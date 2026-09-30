import torch
import numpy as np
from dolfin import VectorFunctionSpace, FunctionSpace, assemble, Constant, Function
from einops import rearrange
from numpy import inner
from tqdm import tqdm
from ufl import Measure


def errors(v, v_ref):
    assert v.shape == v_ref.shape
    assert v.ndim == 3

    diff = v_ref - v
    num = torch.sqrt(torch.sum(((diff**2).sum(dim=-1)),dim=-1))
    sum_norm_ref = torch.sum((v_ref**2).sum(dim=-1),dim=-1)
    denom = torch.sqrt(sum_norm_ref)
    frac = num / denom
    err_mean = torch.mean(frac)
    err_max = torch.max(frac)
    err_rel =  torch.abs(diff) / torch.sqrt(torch.mean(sum_norm_ref))
    return err_mean, err_max, err_rel


def reconstruct_batch_ddpm(ae_traj, params, ddpm, normalizer, device='cpu', return_diff=False, inpaint=0, *args, **kwargs):
    ae_traj = ae_traj.cpu() if isinstance(ae_traj, torch.Tensor) else ae_traj
    ae_traj = normalizer.normalize(ae_traj,'observations')
    traj_length = ae_traj.shape[1]

    inpaint = [inpaint] if not isinstance(inpaint, list) else inpaint
    cond = {i: state for i, state in zip(inpaint, [torch.tensor(ae_traj[:, i], device=device) for i in inpaint])}
    params = torch.tensor(params, device=device)
    if return_diff:
        ae_reconstruct, diff = ddpm.conditional_sample(cond, returns=params, horizon=traj_length, return_diffusion=True, *args, **kwargs)
    else:
        ae_reconstruct = ddpm.conditional_sample(cond, returns=params, horizon=traj_length, *args, **kwargs)

    ae_reconstruct = ae_reconstruct.cpu()
    ae_reconstruct = torch.tensor(normalizer.unnormalize(ae_reconstruct,'observations'),device=device)

    if return_diff:
        return ae_reconstruct, diff
    return ae_reconstruct


def reconstruct_batch_dlrom(ae_traj, params, dts, dlrom, device='cpu'):
    ae_reconstruct = torch.zeros_like(ae_traj, device=device)
    for i, dt in enumerate(dts):
        ae_reconstruct[:,i,:] = dlrom(torch.cat([params, dt.repeat(params.shape[0])[:,None]], dim=1))
    return ae_reconstruct


def reconstruct_batch_1step(ae_traj, params, steprom, device='cpu'):
    ae_reconstruct = torch.zeros_like(ae_traj, device=device)
    ae_reconstruct[:,0] = ae_traj[:,0]
    for i in range(1,ae_traj.shape[1]):
        res = steprom(torch.cat([params, ae_reconstruct[:,i-1,:]], dim=1))
        ae_reconstruct[:,i,:] = res
    return ae_reconstruct


def expand_batch_ae_to_svd(ae_traj, ae):
    with torch.no_grad():
        svd_reconstruct = []
        for traj in ae_traj:
            svd_reconstruct.append(ae.decode(traj))
        svd_reconstruct = torch.stack(svd_reconstruct, dim=0)
    return svd_reconstruct


def expand_batch_svd_to_full(svd_traj, Wx, Wy):
    svd_trunc = Wx.shape[0]
    full_reconstruct_vx = svd_traj[..., :svd_trunc] @ Wx
    full_reconstruct_vy = svd_traj[..., svd_trunc:] @ Wy
    return full_reconstruct_vx, full_reconstruct_vy


class ErrorAccumulator:
    def __init__(self, v_full, v_svd, v_ae, device='cpu'):
        n_traj, n_timestep = v_full.shape[0], v_full.shape[1]
        self.v_full = v_full
        self.v_svd = v_svd
        self.v_ae = v_ae
        self.device = device
        self.sq_norm_diff_acc_full = torch.zeros((n_traj, n_timestep))
        self.sq_norm_ref_acc_full = torch.zeros((n_traj, n_timestep))
        self.sq_norm_diff_acc_svd = torch.zeros((n_traj, n_timestep))
        self.sq_norm_ref_acc_svd = torch.zeros((n_traj, n_timestep))
        self.sq_norm_diff_acc_ae = torch.zeros((n_traj, n_timestep))
        self.sq_norm_ref_acc_ae = torch.zeros((n_traj, n_timestep))

    def accumulate_batch(self, idx, rec_full_vx, rec_full_vy, rec_svd, rec_ae):
        v_combined = torch.stack([rec_full_vx, rec_full_vy], dim=-1)
        v_rec = rearrange(v_combined, "b t n xy -> b t (n xy)")
        # In full order space
        v_ref = self.v_full[idx].to(self.device)
        self.sq_norm_diff_acc_full[idx] += torch.sum((v_ref - v_rec) ** 2, dim=(-1)).cpu()
        #self.sq_norm_ref_acc_full[idx] += torch.sum(v_ref ** 2, dim=(-1)).cpu()
        # On SVD coefficients
        v_ref = self.v_svd[idx].to(self.device)
        self.sq_norm_diff_acc_svd[idx] += torch.sum((v_ref - rec_svd) ** 2, dim=(-1)).cpu()
        #self.sq_norm_ref_acc_svd[idx] += torch.sum(v_ref ** 2, dim=(-1)).cpu()
        # In latent space
        v_ref = self.v_ae[idx].to(self.device)
        self.sq_norm_diff_acc_ae[idx] += torch.sum((v_ref - rec_ae) ** 2, dim=(-1)).cpu()
        #self.sq_norm_ref_acc_ae[idx] += torch.sum(v_ref ** 2, dim=(-1)).cpu()

    def compute_error(self):
        return self.sq_norm_diff_acc_full, self.sq_norm_diff_acc_svd, self.sq_norm_diff_acc_ae
        # err_full = torch.sqrt(self.sq_norm_diff_acc_full / self.sq_norm_ref_acc_full)
        # err_svd = torch.sqrt(self.sq_norm_diff_acc_svd / self.sq_norm_ref_acc_svd)
        # err_ae = torch.sqrt(self.sq_norm_diff_acc_ae / self.sq_norm_ref_acc_ae)
        # return err_full, err_svd, err_ae


def val_ddpm(v_full, v_svd, v_ae, mu, diffusion, normalizer, ae, Wx, Wy, mesh=None, batch_size=10, device='cpu', *args, **kwargs):
    error_acc = ErrorAccumulator(v_full, v_svd, v_ae, device=device)
    with torch.no_grad():
        rec_ae = reconstruct_batch_ddpm(v_ae, mu[:, 0], diffusion, normalizer, device=device, *args, **kwargs)
        return val_prediction(rec_ae, error_acc, ae, Wx, Wy, batch_size, mesh)


def val_ddpm_prob(v_full, v_svd, v_ae, mu, diffusion, normalizer, ae, Wx, Wy, mesh=None, batch_size=10, device='cpu', repeat=10, *args, **kwargs):
        ntraj, ntimesteps = v_full.shape[0], v_full.shape[1]
        err_full_rep = torch.zeros((repeat, ntraj, ntimesteps))
        err_svd_rep = torch.zeros((repeat, ntraj, ntimesteps))
        err_ae_rep = torch.zeros((repeat, ntraj, ntimesteps))
        phys_repeat = []

        for i in range(repeat):
            (err_full, err_svd, err_ae), phys = val_ddpm(v_full, v_svd, v_ae, mu, diffusion, normalizer, ae, Wx, Wy,
                                              mesh=mesh, batch_size=batch_size, device=device, *args, **kwargs)
            phys_repeat.append(phys)
            err_full_rep[i] = err_full
            err_svd_rep[i] = err_svd
            err_ae_rep[i] = err_ae
        return (err_full_rep, err_svd_rep, err_ae_rep), phys_repeat


def val_dlrom(v_full, v_svd, v_ae, mu, dts, dlrom, ae, Wx, Wy, batch_size=10, mesh=None, device='cpu'):
    error_acc = ErrorAccumulator(v_full, v_svd, v_ae, device=device)
    with torch.no_grad():
        rec_ae = reconstruct_batch_dlrom(v_ae, mu[:, 0], dts, dlrom, device=device)
        return val_prediction(rec_ae, error_acc, ae, Wx, Wy, batch_size, mesh)


def val_1step(v_full, v_svd, v_ae, mu, steprom, ae, Wx, Wy, batch_size=10, mesh=None, device='cpu'):
    error_acc = ErrorAccumulator(v_full, v_svd, v_ae, device=device)
    with torch.no_grad():
        rec_ae = reconstruct_batch_1step(v_ae, mu[:, 0], steprom, device=device)
        return val_prediction(rec_ae, error_acc, ae, Wx, Wy, batch_size, mesh)


def val_prediction(pred, error_acc, ae, Wx, Wy, batch_size=10, mesh=None):
    n_traj = pred.shape[0]
    n_batch = int(np.ceil(n_traj / batch_size))
    phys_quant = []

    for batch_idx in range(n_batch):
        idx = range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, n_traj))
        rec_svd = expand_batch_ae_to_svd(pred[idx], ae)
        rec_full_vx, rec_full_vy = expand_batch_svd_to_full(rec_svd, Wx, Wy)
        error_acc.accumulate_batch(idx, rec_full_vx, rec_full_vy, rec_svd, pred[idx])
        del rec_svd
        if mesh is not None:
            phys_quant.append(compute_physical_quantities_trajectories(rec_full_vx, rec_full_vy, mesh))
        del rec_full_vx, rec_full_vy

    if mesh is not None:
        phys_quant = {key: torch.cat([d[key] for d in phys_quant], dim=0) for key in phys_quant[0]}
    return error_acc.compute_error(), phys_quant


def representation_error(v_full, v_svd, v_ae, ae, Wx, Wy, batch_size=10, device='cpu'):
    n_traj = v_full.shape[0]
    n_batch = int(np.ceil(n_traj / batch_size))
    sq_norm_diff_acc_full = torch.zeros((n_traj,), device=device)
    sq_norm_ref_acc_full = torch.zeros((n_traj,), device=device)
    sq_norm_diff_acc_svd = torch.zeros((n_traj,), device=device)
    sq_norm_ref_acc_svd = torch.zeros((n_traj,), device=device)

    for batch_idx in range(n_batch):
        idx = torch.arange(batch_idx * batch_size, min((batch_idx + 1) * batch_size, n_traj))
        best_svd = expand_batch_ae_to_svd(v_ae[idx].to(device), ae)
        best_full_vx, best_full_vy = expand_batch_svd_to_full(best_svd, Wx, Wy)
        v_combined = torch.stack([best_full_vx, best_full_vy], dim=-1)
        v_rec = rearrange(v_combined, "b t n xy -> b t (n xy)")
        # In full order space
        v_ref = v_full[idx].to(device)
        sq_norm_diff_acc_full[idx] += torch.sum((v_ref - v_rec)**2, dim=(-1,-2))
        sq_norm_ref_acc_full[idx] += torch.sum(v_ref**2, dim=(-1,-2))
        # On SVD coefficients
        v_ref = v_svd[idx].to(device)
        sq_norm_diff_acc_svd[idx] += torch.sum((v_ref - best_svd)**2, dim=(-1,-2))
        sq_norm_ref_acc_svd[idx] += torch.sum(v_ref**2, dim=(-1,-2))
    err_full = torch.sqrt(sq_norm_diff_acc_full / sq_norm_ref_acc_full)
    err_svd = torch.sqrt(sq_norm_diff_acc_svd / sq_norm_ref_acc_svd)
    return err_full, err_svd


def compute_physical_quantities_trajectories(vx, vy, mesh):
    # vx, vy are of shape (ntraj, t, ndof)
    v = rearrange([vx, vy], 'xy ntraj t ndof -> ntraj t (ndof xy)')

    dx = Measure("dx", domain=mesh)
    Vh = VectorFunctionSpace(mesh, "CG", 2)
    S = FunctionSpace(mesh, "CG", 1)
    e_kin = torch.zeros(v.shape[:-1])
    #div_u = torch.zeros(v.shape[:-1])
    area = assemble(Constant(1.0)*dx)
    for i, traj in tqdm(enumerate(v)):
        for j, vt in enumerate(traj):
            u = Function(Vh)
            vt = vt.cpu().numpy().astype(np.float64) if isinstance(vt, torch.Tensor) else vt
            u.vector().set_local(vt)
            u.vector().apply("insert")
            res = assemble(0.5 * inner(u,u) * dx)
            e_kin[i,j] = torch.tensor(res, dtype=torch.float32)
            #div_u_field = project(div(u), S)
            #div_u[i,j] = assemble(abs(div_u_field) * dx) / area
    #return {'kinetic': e_kin, 'divergence': div_u}
    return {'kinetic': e_kin}
