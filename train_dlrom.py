import argparse
import os

import einops
import h5py
import numpy as np
import torch

from models import AEMLP, ConvAE, AE_INN, DLRom, AEIdentity
from memory_buffer import FlowReplayBuffer as FlowDataLoader, ReplayBuffer
from memory_buffer import ReplayBuffer as KSDataLoader
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, choices=['fluid', 'ks'], default='fluid')
    parser.add_argument('--train_data', type=str, default='flowpastobject_2d/FlowAroundObstacle_data_10sec/svd/train/256.npz')
    parser.add_argument('--val_data', type=str, default='flowpastobject_2d/FlowAroundObstacle_data_10sec/svd/val/256.npz')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--num-epochs', type=int, default=1500,
                        help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--reg-coefficient', type=float, default=0.0,
                        help='L2 regularization coefficient.')
    parser.add_argument('--training', default=True,
                        help='Train the models.')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Number of hidden units in MLPs.')
    parser.add_argument('--svd-truncation', type=int, default=256,
                        help='SVD truncation.')
    parser.add_argument('--latent-dim', nargs="+",type=int,
                        help='Latent dimension of the autoencoder.')
    parser.add_argument('--experiment', type=str, default='cylinderflow',
                        help='Experiment.')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='How many batches to wait before saving')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1).')
    parser.add_argument('--load', default=False,
                        help='Load trained model.')

    # Parser for innae
    subparsers = parser.add_subparsers(required=True, dest='AE_type', help='AE type.')
    parser_id = subparsers.add_parser('identity', help='Do not reduce.')
    parser_mpl = subparsers.add_parser('mlp', help='Use MLP architecture.')
    parser_conv = subparsers.add_parser('conv', help='Use Conv architecture.')
    parser_innae = subparsers.add_parser('innae', help='Use INNAE architecture.')
    parser_innae.add_argument('--model-type', type=str, default='inn_mlpAE_layers_spectral',
                        help='Model type.')
    parser_innae.add_argument('--activation', type=str, default='gelu',
                        help='Activation function.')
    parser_innae.add_argument('--layer-type', type=str, default='rev_multiplicative_layer',
                        help='Layer type (rev_layer or rev_multiplicative_layer).')
    parser_innae.add_argument('--spectral-norm', type=bool, default=True,
                        help='Use spectral normalization.')
    parser_innae.add_argument('--zero-pad-coef', type=float, default=0.0,  # 1.0
                        help='Coefficient of the zero pad loss.')
    parser_innae.add_argument('--lip-reg-coef', type=float, default=0.0,  # 1e-4
                        help='Coefficient of the Lipschitz regularization loss.')

    args = parser.parse_args()
    return args


def extract_snapshot_mu_dt(batch, device='cpu'):
    if 'vx_snapshot' in batch and 'vy_snapshot' in batch:
        vx_snapshots = torch.from_numpy(batch['vx_snapshots']).to(device)
        vy_snapshots = torch.from_numpy(batch['vy_snapshots']).to(device)
        snapshots = torch.cat([vx_snapshots, vy_snapshots], dim=1)
    elif 'snapshots' in batch:
        snapshots = torch.from_numpy(batch['snapshots']).to(device)
    else:
        raise ValueError("Batch type not supported")
    mus = torch.from_numpy(batch['mus']).to(device)
    dts = torch.from_numpy(batch['dts']).to(device)
    dts = dts if dts.dim() == 2 else dts[:, None]
    return snapshots, mus, dts


def eval_batch(model, rom, batch, device='cpu', validation=False):
    if validation:
        model.eval()
        rom.eval()
    snapshots, mus, dts = extract_snapshot_mu_dt(batch, device)

    recon_snapshots, z = model(snapshots)
    zn = rom(torch.cat([mus, dts], dim=1))
    recon_snapshots_n = model.decode(zn)

    loss_recon_z = torch.nn.functional.mse_loss(z, zn)
    loss_recon_n = torch.nn.functional.mse_loss(snapshots, recon_snapshots_n)

    loss = loss_recon_n + loss_recon_z
    # print("loss reconstruction dlrom = {}".format(loss_recon_n.item()))
    # print("loss regularization z-zn = {}".format(loss_recon_z.item()))

    projection_error = torch.sqrt( torch.square(torch.norm(snapshots - recon_snapshots)) / (torch.square(torch.norm(snapshots))))

    return loss, projection_error, recon_snapshots_n, zn

def train_epoch(model, rom, data_loader, optimizer, device='cpu'):
    train_loss = 0
    proj_error = 0

    for _, batch in data_loader:
        loss, projection_error, _, _ = eval_batch(model, rom, batch, device=device)
        train_loss += loss.item()
        proj_error += projection_error.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return train_loss / (data_loader.size/data_loader.batch_size), proj_error / (data_loader.size/data_loader.batch_size)


def train(model, rom, train_loader, validate, optimizer, patience=500, max_epoch=1000, save_dir="", log_interval=5, device='cpu'):
    init_patience = patience
    latest_epoch = 0
    logs = {
        'train_loss': [],
        'train_proj_error': [],
    }
    for epoch in range(max_epoch):
        print('############################################################')
        latest_epoch = epoch # For saving final model, possibly after early stopping
        model.train()
        rom.train()
        train_loss, train_proj_error = train_epoch(model, rom, train_loader, optimizer, device=device)
        logs['train_loss'].append(train_loss)
        logs['train_proj_error'].append(train_proj_error)
        print('====> Epoch: {} Average training loss: {:.10f}'.format(epoch, train_loss))
        print('====> Epoch: {} Average training projection error: {:.10f}'.format(epoch, train_proj_error))

        if epoch % log_interval == 0:
            torch.save({'model': model.state_dict(),'epoch': epoch, 'logs': logs}, os.path.join(save_dir, 'AE_model_latest.pth'))
            torch.save({'rom': rom.state_dict(), 'epoch': epoch, 'logs': logs}, os.path.join(save_dir, 'dlrom_latest.pth'))

        model.eval()
        rom.eval()
        with torch.no_grad():
            logs, new_best, save_files = validate(model, rom, logs, device=device)

            for save_file in save_files:
                file, ext = os.path.splitext(save_file)
                save_file_ae = f"{file}_dlrom_ae{ext}"
                save_file_rom = f"{file}_dlrom{ext}"
                torch.save({'model': model.state_dict(),'epoch': epoch, 'logs': logs}, os.path.join(save_dir, save_file_ae))
                torch.save({'rom': rom.state_dict(), 'epoch': epoch, 'logs': logs}, os.path.join(save_dir, save_file_rom))

            for metric in logs:
                print(f'====> Epoch: {epoch} {metric}: {logs[metric][-1]:.10f}')

            if new_best:
                patience = init_patience
            else:
                patience -= 1
            if patience == 0:
                print(f"Early stopping at epoch {epoch}")
                break
            print(f'Patience: {patience}')
    print('Training finished.')
    torch.save({'model': model.state_dict(),'epoch': latest_epoch, 'logs': logs}, os.path.join(save_dir, 'AE_model_latest.pth'))
    torch.save({'rom': rom.state_dict(), 'epoch': latest_epoch, 'logs': logs}, os.path.join(save_dir, 'dlrom_latest.pth'))
    return logs


class FlowValidator:
    def __init__(self, svd_data, full_data, mus, svd, full_batch=2000):
        # vx = svd_data[0]
        # vy = svd_data[1]
        #
        # vx = einops.rearrange(vx, 'b t svd -> (b t) svd')
        # vy = einops.rearrange(vy, 'b t svd -> (b t) svd')
        # self.svd_data = {'vx_snapshots': vx,
        #                  'vy_snapshots': vy,}

        dt = 0.05
        T = 10.0
        ntimesteps = round(T / dt)
        timesteps = np.arange(0, int(T), step=dt)
        ntrajectories = full_data.shape[0]
        timesteps = timesteps.reshape(1, ntimesteps)
        timesteps = timesteps.repeat(ntrajectories, axis=0).reshape(ntrajectories*ntimesteps, -1).astype(np.float32)

        self.vx_full = torch.tensor(full_data[:, :, 0:full_data.shape[-1]:2])
        self.vy_full = torch.tensor(full_data[:, :, 1:full_data.shape[-1]:2])

        self.svd_Wx = svd['svd_vx'][2]
        self.svd_Wy = svd['svd_vy'][2]

        vx_svd = self.vx_full @ torch.tensor(self.svd_Wx.T)
        vx_svd = einops.rearrange(vx_svd, 'b t n -> (b t) n').numpy()
        vy_svd = self.vy_full @ torch.tensor(self.svd_Wy.T)
        vy_svd = einops.rearrange(vy_svd, 'b t n -> (b t) n').numpy()

        self.v_full = torch.cat((self.vx_full, self.vy_full), dim=-1)
        self.v_full = einops.rearrange(self.v_full, 'b t n -> (b t) n')
        self.svd = svd
        self.svd_dim = len(svd['svd_vx'][1])
        self.n_val = ntrajectories
        self.full_batch = full_batch
        self.n_batch = self.n_val / full_batch

        self.mus = torch.tensor(mus)
        self.mus = einops.rearrange(self.mus, 'b t n -> (b t) n').numpy()
        self.dts = timesteps

        self.svd_data = {'vx_snapshots': vx_svd,
                         'vy_snapshots': vy_svd,
                         'mus': self.mus,
                         'dts': self.dts}

    def __call__(self, model, rom, logs, device='cpu'):
        new_best = False
        save_files = []
        logs = {} if logs is None else logs
        for metric in ['val_loss_svd', 'val_proj_error_svd', 'val_loss_v', 'val_proj_error_v']:
            if metric not in logs:
                logs[metric] = []
        svd_Wx = torch.tensor(self.svd_Wx).to(device)
        svd_Wy = torch.tensor(self.svd_Wy).to(device)

        val_loss_svd, val_proj_error_svd, recon_svd, _ = eval_batch(model, rom, self.svd_data, device, validation=True)

        val_loss_v = 0
        val_proj_error_v = 0
        for batch in range(int(np.ceil(self.n_batch))):
            batch_idx = range(batch * self.full_batch, (batch + 1) * self.full_batch)

            Vx_recon = recon_svd[batch_idx, :self.svd_dim] @ svd_Wx
            Vy_recon = recon_svd[batch_idx, self.svd_dim:] @ svd_Wy
            V_recon = torch.cat([Vx_recon, Vy_recon], dim=-1)
            v_full_batch = self.v_full[batch_idx].to(device)
            val_loss_v += torch.nn.functional.mse_loss(v_full_batch, V_recon).item()
            val_proj_error_v += 0.5 * torch.sqrt(
                torch.square(torch.norm(v_full_batch - V_recon)) / (torch.square(torch.norm(v_full_batch)))).item()
        val_loss_v = val_loss_v / self.n_batch
        val_proj_error_v = val_proj_error_v / self.n_batch
        if len(logs['val_proj_error_v']) == 0 or val_proj_error_v < min(logs['val_proj_error_v']):
            print("New best projection on V found")
            save_files.append('AE_best_proj_model_v.pth')
        if len(logs['val_loss_v']) == 0 or val_loss_v < min(logs['val_loss_v']):
            print("New best validation loss on V found")
            save_files.append('AE_best_val_loss_v.pth')
        if len(logs['val_proj_error_svd']) == 0 or val_proj_error_svd < min(logs['val_proj_error_svd']):
            print("New best projection on SVD found")
            save_files.append('AE_best_proj_model_svd.pth')
        if len(logs['val_loss_svd']) == 0 or val_loss_svd < min(logs['val_loss_svd']):
            print("New best validation loss on SVD found")
            save_files.append('AE_best_val_loss_svd.pth')
            new_best = True

        logs['val_loss_svd'].append(val_loss_svd.item())
        logs['val_proj_error_svd'].append(val_proj_error_svd.item())
        logs['val_loss_v'].append(val_loss_v)
        logs['val_proj_error_v'].append(val_proj_error_v)
        return logs, new_best, save_files


class KSValidator:
    def __init__(self, val_data):
        input_dim = val_data['obs'].shape[-1]
        nval_traj = val_data['obs'].shape[0]
        mu_dim = val_data['mu'].shape[-1]
        ntimesteps = val_data['dts'].shape[0]
        self.dataloader = ReplayBuffer(
            input_dim=input_dim,
            mu_dim=mu_dim,
            timestep=ntimesteps,
            size=nval_traj,
        )

        for i in range(nval_traj):
            for j in range(ntimesteps):
                self.dataloader.store(val_data['obs'][i, j], val_data['mu'][i, j], val_data['dts'][j])

    def __call__(self, model, rom, logs, device='cpu'):
        new_best = False
        save_files = []
        logs = {} if logs is None else logs
        for metric in ['val_loss', 'val_proj_error']:
            if metric not in logs:
                logs[metric] = []

        val_loss, val_proj_error, recon, _ = eval_batch(model, rom, self.dataloader.get_all_samples(), device, validation=True)

        if len(logs['val_loss']) == 0 or val_loss < min(logs['val_loss']):
            print("New best validation loss found")
            save_files.append('AE_best_val_loss.pth')
            new_best = True
        if len(logs['val_proj_error']) == 0 or val_proj_error < min(logs['val_proj_error']):
            print("New best projection found")
            save_files.append('AE_best_proj.pth')
            new_best = True

        logs['val_loss'].append(val_loss.item())
        logs['val_proj_error'].append(val_proj_error.item())

        return logs, new_best, save_files


def main():
    args = parse_args()
    print(f"Running with args: {args}")
    set_seed(args.seed)
    device = 'cpu'
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = 'cuda'

    # Optimization settings
    lr = args.learning_rate
    reg_coef = args.reg_coefficient

    # AE settings
    ae_type = args.AE_type
    svd_trunc = args.svd_truncation
    h_dim = args.hidden_dim
    latent_dims = args.latent_dim

    # Training settings
    log_interval = args.log_interval
    batch_size = args.batch_size

    # Train data
    train_data_file = args.train_data
    train_data = np.load(train_data_file, allow_pickle=True)


    # Validation data
    val_data_svd_file = args.val_data
    val_data_svd = np.load(val_data_svd_file, allow_pickle=True)

    # Data properties
    if args.data_type == 'ks':
        xdim = train_data['obs'].shape[-1]
        ntimesteps = train_data['obs'].shape[-2]
        ntrain = train_data['obs'].shape[0]
        nparams = train_data['mu'].shape[-1]
        train_loader = KSDataLoader(input_dim=xdim, timestep=ntimesteps, size=ntrain, mu_dim=nparams, batch_size=batch_size)
        for i in range(ntrain):
            for j in range(ntimesteps):
                train_loader.store(train_data['obs'][i,j], train_data['mu'][i, j], train_data['dts'][j])
        validate = KSValidator(val_data_svd)
    else:
        dt = 0.05
        T = 10.0
        ntimesteps = round(T / dt)
        timesteps = np.arange(0, int(T), step=dt)
        nparams = 4
        ntrain = train_data['v'][0].shape[0]
        svd = val_data_svd['svd'][()]
        val_data_svd = val_data_svd['v']
        xdim = 2 * svd_trunc

        full_data_file = 'flowpastobject_2d/FlowAroundObstacle_data_10sec/full/val/FlowAroundObstacle_data_10sec.hdf5'
        data = h5py.File(full_data_file, 'r')
        val_data_v = data['V']
        val_data_mu = data["MU"]

        train_loader = FlowDataLoader(input_dim=xdim, timestep=ntimesteps, size=ntrain, mu_dim=nparams, batch_size=batch_size)
        train_v = train_data['v']
        train_mu = train_data['mu']
        for i in range(ntrain):
            for j in range(ntimesteps):
                train_loader.store(train_v[0, i, j], train_v[1, i, j], train_mu[i, j], timesteps[j])

        validate = FlowValidator(val_data_svd, val_data_v, val_data_mu, svd)

    latent_dims = [1, 2, 3, 4, 5, 10, 20, 50, 100, 150] if latent_dims is None else latent_dims

    for latent_dim in latent_dims:
        if ae_type == 'mlp':
            model = AEMLP(x_dim=xdim, z_dim=latent_dim, h_dim=h_dim, mu_dim=nparams)
        elif ae_type == 'conv':
            model = ConvAE(x_dim=xdim, z_dim=latent_dim, h_dim=h_dim, mu_dim=nparams)
        elif ae_type == 'innae':
            model = AE_INN(x_dim=xdim, z_dim=latent_dim, h_dim=h_dim, activation=args.activation, inn_layer_type=args.layer_type,
               use_spectral_norm=args.spectral_norm)
        elif ae_type == 'identity':
            model = AEIdentity()
        else:
            raise ValueError('Unknown ae_type')
        model.to(device)

        rom = DLRom(z_dim=latent_dim, mu_dim=nparams, h_dim=h_dim).to(device)

        optimizer = torch.optim.AdamW([
            {'params': model.parameters()},
            {'params': rom.parameters()},
            ], lr=lr, weight_decay=reg_coef,  betas=(0.9, 0.98))

        directory = os.path.dirname(os.path.abspath(__file__))
        save_dir = directory + '/results/' + str(args.experiment) + '/' + str(ae_type) + '_dlrom/latent_dim_' + str(latent_dim)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f'Training {ae_type}-dlrom with latent dimension: {latent_dim}')
        train(model, rom, train_loader, validate, optimizer, patience=100, max_epoch=args.num_epochs, save_dir=save_dir, log_interval=log_interval, device=device)

if __name__ == '__main__':
    main()