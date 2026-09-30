import argparse

import numpy as np
from dolfin import VectorFunctionSpace, Function, Mesh, vertex_to_dof_map, \
    ALE

from errors import expand_batch_ae_to_svd, expand_batch_svd_to_full
from inspect_rom import load_test_data, next_model, load_rom_errors, save_rom_errors

from scipy.interpolate import RBFInterpolator


def batched_compute_physical_quantities(predict, ae, Wx, Wy, mesh, ntraj, batchsize=10):
    phys = []
    for i in range(int(np.ceil(ntraj / batchsize))):
        batch_idx = slice(i * batchsize, min((i + 1) * batchsize, ntraj))
        svd_predict = expand_batch_ae_to_svd(predict[batch_idx], ae)
        full_predict_vx, full_predict_vy = expand_batch_svd_to_full(svd_predict, Wx, Wy)
        del svd_predict
        phys.append(compute_physical_quantities_trajectories(full_predict_vx, full_predict_vy, mesh))
        del full_predict_vx, full_predict_vy
    return {key: torch.cat([d[key] for d in phys], dim=0) for key in phys[0]}


def predict_phys_ddpm(v_0, params, ddpm, normalizer, ae, n_steps, Wx, Wy, mesh, device='cpu', batchsize=10):
    v_0 = normalizer.normalize(v_0.cpu(), 'observations')

    cond = {0: torch.from_numpy(v_0[:,0]).to(device)}
    params = torch.tensor(params, device=device)
    with torch.no_grad():
        predict = ddpm.conditional_sample(cond, returns=params, horizon=n_steps)
        predict.cpu()
        predict = torch.tensor(normalizer.unnormalize(predict.cpu(), 'observations'), device=device)
        return batched_compute_physical_quantities(predict, ae, Wx, Wy, mesh, v_0.shape[0], batchsize=batchsize)


def predict_phys_ddpm_prob(v_0, params, ddpm, normalizer, ae, n_steps, Wx, Wy, mesh, repeat=1, device='cpu'):
    phys = []
    for i in range(repeat):
        phys.append(predict_phys_ddpm(v_0, params, ddpm, normalizer, ae, n_steps, Wx, Wy, mesh, device=device))
    phys_stats = {}
    for quantity in phys[0]:
        phys_stack = torch.stack([p[quantity] for p in phys])
        phys_stats[quantity] = (phys_stack.mean(dim=0), phys_stack.std(dim=0))
    return phys_stats


def predict_phys_dlrom(v_0, params, dlrom, ae, timesteps, Wx, Wy, mesh, device='cpu', batchsize=10):
    shape = (v_0.shape[0], len(timesteps), v_0.shape[-1])
    predict = torch.zeros(shape, device=device)
    with torch.no_grad():
        for i, dt in enumerate(timesteps):
            predict[:, i, :] = dlrom(torch.cat([params, dt.repeat(params.shape[0])[:, None]], dim=1))
        return batched_compute_physical_quantities(predict, ae, Wx, Wy, mesh, shape[0], batchsize=batchsize)


def predict_phys_1step(v_0, params, steprom, ae, n_steps, Wx, Wy, mesh, ref=None, device='cpu', batchsize=10):
    shape = (v_0.shape[0], n_steps, v_0.shape[-1])
    predict = torch.zeros(shape, device=device)
    predict[:,0,:] = v_0[:,0,:]
    with torch.no_grad():
        for i in range(1,n_steps):
            res = steprom(torch.cat([params, predict[:, i - 1, :]], dim=1))
            predict[:, i, :] = res
        return batched_compute_physical_quantities(predict, ae, Wx, Wy, mesh, shape[0], batchsize=batchsize)




def parse_args():
    parser = argparse.ArgumentParser(description="Override default model configurations")

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["ddpm", "dlrom", "1step"],
        default=["ddpm", "dlrom", "1step"],
        help="List of methods to include"
    )

    parser.add_argument(
        "--ae_types",
        nargs="+",
        choices=["mlp", "conv", "innae"],
        default=["mlp", "conv", "innae"],
        help="List of autoencoder types"
    )

    parser.add_argument(
        "--ae_dyns",
        nargs="+",
        choices=["None", "1step", "dlrom"],
        default=["None", "1step", "dlrom"],
        help="List of autoencoder dynamics (use 'None' as string)"
    )

    parser.add_argument(
        "--latent_dims",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5, 10, 20, 50],
        default=[1, 2, 3, 4, 5, 10, 20, 50],
        help="List of latent dimensions"
    )

    parser.add_argument(
        "--compute_baseline",
        type=bool,
        default=False,
        help="Compute energy in test dataset"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Convert "None" string back to actual None
    ae_dyns = [None if d == "None" else d for d in args.ae_dyns]

    methods = args.methods
    ae_types = args.ae_types
    latent_dims = args.latent_dims
    ae_reg = [True]
    compute_baseline = args.compute_baseline
    print("Methods:", methods)
    print("AE types:", ae_types)
    print("AE dynamics:", ae_dyns)
    print("Latent dims:", latent_dims)
    print("Compute baseline:", compute_baseline)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dt = 0.05
    T = 20.0
    ntimesteps = round(T / dt)
    timesteps = torch.arange(0, int(T), step=dt, device=device)[:, None]
    mu_dim = 4
    mesh = mesh_generator(0.25, 0.25)

    v_full, v_svd_np, mu, Wx, Wy = load_test_data()
    v_full = torch.from_numpy(v_full[:])
    v_full = rearrange(v_full, "b t (n xy) -> xy b t n", xy=2)

    if compute_baseline:
        physics_test_data = compute_physical_quantities_trajectories(v_full[0], v_full[1], mesh)
        with open("test_physical_quantities.pickle", "wb") as f:
            pickle.dump(physics_test_data, f)

    #v_svd = torch.from_numpy(v_svd_np[:])
    #v_svd = rearrange(v_svd, "xy b t n -> b t (xy n)")
    mu = torch.from_numpy(mu[:,0,:]).float().to(device)
    filename = "rom_extrapolate.pickle"
    phys_quant = load_rom_errors(filename=filename)
    computed_configs = []
    for method, ae_t, reg, lat_dim, ae, dyn, ae_dyn in next_model(methods, ae_types, ae_reg, latent_dims, ae_dyns):
        print(f'Starting {method}.{ae_t}.{ae_dyn}.{reg}:{lat_dim}')
        computed_configs.append((method, ae_t, reg, lat_dim, ae_dyn))
        ae.eval()
        v_ae = ae_encode_dataset(v_svd_np[:,:,[0],:], None, ae, device=device)
        if method == 'ddpm':
            diffusion, normalizer = dyn
            diffusion.eval()
            pred_energy = predict_phys_ddpm_prob(v_ae, mu, diffusion, normalizer, ae, ntimesteps, Wx, Wy, mesh, device=device, repeat=10)
            phys_quant[method][ae_t][reg][ae_dyn][lat_dim] = pred_energy
            continue
        elif method == 'dlrom':
            dyn.eval()
            pred_energy = predict_phys_dlrom(v_ae, mu, dyn, ae, timesteps, Wx, Wy, mesh, device=device)
        elif method == '1step':
            dyn.eval()
            pred_energy = predict_phys_1step(v_ae, mu, dyn, ae, ntimesteps, Wx, Wy, mesh, ref=v_full, device=device)
        else:
            raise ValueError(f"Unknown method {method}")
        phys_quant[method][ae_t][reg][lat_dim] = pred_energy
    save_rom_errors(phys_quant, filename=filename)
    print('Computed configs:')
    for method, ae_t, reg, lat_dim, ae_dyn in computed_configs:
        print(f"{method}.{ae_t}.{ae_dyn}.{reg}:{lat_dim}")
