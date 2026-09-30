import argparse
import glob
import os
import yaml

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import randomized_svd
from einops import rearrange, asnumpy

from models import AEMLP, ConvAE, AE_INN
from utils import load_trajectories


def compute_svd_and_scale(Vdata, trunc_dim, scale=False):
    ntraj, ntsteps, nvelocity = Vdata.shape
    Vx = Vdata[:, :, 0:nvelocity:2]
    Vy = Vdata[:, :, 1:nvelocity:2]
    del Vdata

    Vx_train = np.ascontiguousarray(Vx.reshape(-1, nvelocity // 2))
    Vy_train = np.ascontiguousarray(Vy.reshape(-1, nvelocity // 2))

    # Compute svd on train samples
    print("Computing SVD...")
    Ux, Sx, Wx = randomized_svd(Vx_train, n_components=trunc_dim)
    Uy, Sy, Wy = randomized_svd(Vy_train, n_components=trunc_dim)
    svd = {"svd_vx": (Ux, Sx, Wx), "svd_vy": (Uy, Sy, Wy)}

    scaler = None
    if scale:
        # Compute scaling on train samples
        scalerVx = MinMaxScaler()
        scalerVy = MinMaxScaler()
        print("Scaling...")
        scalerVx.fit(Vx_train @ Wx.transpose())
        scalerVy.fit(Vy_train @ Wx.transpose())
        scaler = (scalerVx, scalerVy)
    return svd, scaler


def svd_encode_dataset(Vdata, mu, svd, scaler=None, savefile=None):
    ntraj, ntsteps, nvelocity = Vdata.shape
    Vx = Vdata[:, :, 0:nvelocity:2]
    Vy = Vdata[:, :, 1:nvelocity:2]
    del Vdata

    # Apply svd to dataset
    Wx = svd["svd_vx"][2]
    Wy = svd["svd_vy"][2]
    print("Applying Reduction...")
    Vx_svd = np.ascontiguousarray(rearrange(Vx,"b t p -> (b t) p")) @ Wx.transpose()
    Vy_svd = np.ascontiguousarray(rearrange(Vy,"b t p -> (b t) p")) @ Wy.transpose()

    if scaler is not None:
        scalerVx, scalerVy = scaler
        # Apply scaling to dataset
        Vx_svd = scalerVx.transform(Vx_svd)
        Vy_svd = scalerVy.transform(Vy_svd)

    v = rearrange([Vx_svd, Vy_svd], "xy (b t) r -> xy b t r", b=ntraj)
    # Save svd and scaled dataset
    if savefile is not None:
        print("Saving Reduced data...")
        np.savez_compressed(savefile, v=v, mu=mu, svd=svd, scaler=scaler)
        print(f"\tSVD reduced data saved as {savefile}.")
    return v, svd, scaler


def ae_encode_dataset(v, mu, ae, savefile=None, device="cpu", ae_file=None, svd=None, scaler=None, batch=100):
    ntraj = v.shape[1]
    v = rearrange(v,"xy b t r -> (b t) (xy r)")
    # Apply ae to dataset
    latents = []
    ae.eval()
    with torch.no_grad():
        for i in range(int(np.ceil(v.shape[0] / batch))):
            idx_batch = range(i * batch, min((i + 1) * batch, v.shape[0]))
            v_batch = v[idx_batch]
            v_batch = v_batch.to(device) if isinstance(v_batch, torch.Tensor) else torch.from_numpy(v_batch).to(device)
            v_latent_batch = ae.encode(v_batch)
            latents.append(v_latent_batch)
        v_latent = torch.cat(latents)
    v_latent = rearrange(v_latent, "(b t) r -> b t r", b=ntraj)

    if savefile is not None:
        # Save ae_dataset
        print(f"Saving AE data...")
        np.savez_compressed(savefile, v=asnumpy(v_latent), mu=mu, ae_file=ae_file, svd=svd, scaler=scaler)
        print(f"\tAE reduced data saved as {savefile}.")
    return v_latent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument('--svd', action='store_true')
    parser.add_argument('--svd_trunc', type=int, default=256)
    parser.add_argument('--ae', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--latent_dim', type=int, default=20)
    parser.add_argument('--h_dim', type=int, default=512)

    # Parser for innae
    subparsers = parser.add_subparsers(required=False, dest='AE_type', help='AE type.')
    parser_mpl = subparsers.add_parser('mlp', help='Use MLP architecture.')
    parser_conv = subparsers.add_parser('conv', help='Use Conv architecture.')
    parser_innae = subparsers.add_parser('innae', help='Use INNAE architecture.')
    #parser_innae.add_argument('--model-type', type=str, default='inn_mlpAE_layers_spectral',
    #                    help='Model type.')
    parser_innae.add_argument('--activation', type=str, default='gelu',
                        help='Activation function.')
    parser_innae.add_argument('--layer-type', type=str, default='rev_multiplicative_layer',
                        help='Layer type (rev_layer or rev_multiplicative_layer).')
    parser_innae.add_argument('--spectral-norm', type=bool, default=True,
                        help='Use spectral normalization.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(0)
    torch.manual_seed(0)

    compute_svd = args.svd
    compute_ae = args.ae is not None
    # Load dataset
    data_dir = args.data #"FlowAroundObstacle_data_10sec.hdf5"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(data_dir, "svd", split))
            os.makedirs(os.path.join(data_dir, "ae", split))

    svd_trunc = args.svd_trunc
    scale = False

    if compute_svd:
        print("Loading data...")
        file = glob.glob(os.path.join(data_dir, "full", "*.hdf5"))[0]
        print(f"Found data file: {file}")
        V, _, mu = load_trajectories(file)
        with open('idx_1000.yml', 'rb') as handle:
            idx = yaml.safe_load(handle)

        svd, scaler = compute_svd_and_scale(V[idx['train']], trunc_dim=svd_trunc, scale=scale)

        for split in ['train', 'val', 'test']:
            svd_encode_dataset(V[idx[split]], mu[idx[split]], svd, scaler, savefile=os.path.join(file, "svd", split, str(svd_trunc)))

    # Load ae
    if compute_ae:
        print("Loading AE...")
        ae_file = args.ae
        latent_dim = args.latent_dim
        h_dim = args.h_dim
        nparams = 4
        ae_type = args.AE_type

        if ae_type == 'mlp':
            model = AEMLP(x_dim=2 * svd_trunc, z_dim=latent_dim, h_dim=h_dim, mu_dim=nparams)
        elif ae_type == 'conv':
            model = ConvAE(x_dim=2 * svd_trunc, z_dim=latent_dim, h_dim=h_dim, mu_dim=nparams)
        elif ae_type == 'innae':
            activation = args.activation
            spectral_norm = args.spectral_norm
            layer_type = args.layer_type
            model = AE_INN(x_dim=2 * svd_trunc, z_dim=latent_dim, h_dim=h_dim, activation=activation,
                           inn_layer_type=layer_type, use_spectral_norm=spectral_norm)
        else:
            raise ValueError(f"Unrecognized AE_type {ae_type}.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        weights = torch.load(ae_file,map_location=device)
        model.load_state_dict(weights['model'])
        model = model.to(device)

        filename_ae_conf = os.path.join(data_dir, 'ae', args.name, ae_type + str(latent_dim) + '.pickle')
        model.save_config(filename_ae_conf)

        for split in ['train', 'val', 'test']:
            try:
                data = np.load(os.path.join(data_dir, "svd", split, str(svd_trunc) + '.npz'), allow_pickle=True)
            except FileNotFoundError:
                print(f"No {split} data file found.")
                continue
            v = data['v']
            mu = data['mu']
            svd = data['svd']
            scaler = data['scaler']
            os.makedirs(os.path.join(data_dir, 'ae', args.name, split), exist_ok=True)
            filename_ae = os.path.join(data_dir, 'ae', args.name, split, ae_type + str(latent_dim) + '.npz')
            v_ae = ae_encode_dataset(v, mu, model, savefile=filename_ae, device=device, ae_file=ae_file, svd=svd, scaler=scaler)
