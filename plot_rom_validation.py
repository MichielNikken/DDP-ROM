import pickle
from typing import Dict, Iterable, Optional, Tuple, List, Any

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from inspect_rom import load_rom_errors


def _is_lat_level_dict(d: Dict) -> bool:
    """Return True if dict keys look like latent-dimension keys (integers or integer-strings)."""
    if not isinstance(d, dict) or not d:
        return False
    for k in d.keys():
        try:
            int(k)
        except Exception:
            return False
    return True


def _iter_selected_entries(
    results: Dict,
    methods: Optional[Iterable[str]] = None,
    aes: Optional[Iterable[str]] = None,
    regs: Optional[Iterable[bool]] = None,
    lat_dims: Optional[Iterable[int]] = None,
    ae_dyns: Optional[Iterable[Any]] = None,
):
    """
    Yield tuples (method, ae, reg, lat_dim, value) for entries matching selectors.

    - Supports optional ae_dyn level under results[method][ae][reg] (used by ddpm).
    - If ae_dyns is provided, it filters ddpm entries by that set.
    """
    methods_sel = set(methods) if methods is not None else None
    aes_sel = set(aes) if aes is not None else None
    regs_sel = set(regs) if regs is not None else None
    lat_sel = set(lat_dims) if lat_dims is not None else None
    ae_dyns_sel = set(ae_dyns) if ae_dyns is not None else None

    for method, mdict in results.items():
        if methods_sel is not None and method not in methods_sel:
            continue
        if not isinstance(mdict, dict):
            continue
        for ae, aedict in mdict.items():
            if aes_sel is not None and ae not in aes_sel:
                continue
            if not isinstance(aedict, dict):
                continue
            for reg, rdict in aedict.items():
                if regs_sel is not None and reg not in regs_sel:
                    continue
                # rdict may be:
                #  - lat-level dict: {lat_dim: value, ...}  (non-ddpm)
                #  - ae_dyn-level dict: {ae_dyn: {lat_dim: value, ...}, ...} (ddpm)
                if _is_lat_level_dict(rdict):
                    # direct lat-level
                    for lat_dim, value in rdict.items():
                        lat_int = int(lat_dim)
                        if lat_sel is not None and lat_int not in lat_sel:
                            continue
                        yield method, ae, reg, None, lat_int, value
                else:
                    # assume an intermediate ae_dyn level (or other nested dicts)
                    for ae_dyn, inner in rdict.items():
                        # filter ae_dyn if requested
                        if ae_dyns_sel is not None and ae_dyn not in ae_dyns_sel:
                            continue
                        if not isinstance(inner, dict):
                            continue
                        if not _is_lat_level_dict(inner):
                            # if inner is not lat-level, skip
                            continue
                        for lat_dim, value in inner.items():
                            lat_int = int(lat_dim)
                            if lat_sel is not None and lat_int not in lat_sel:
                                continue
                            yield method, ae, reg, ae_dyn, lat_int, value


# -------------------------
# Utility: convert torch or array-like to numpy
# -------------------------
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(float)
    else:
        return np.asarray(x, dtype=float)

def _aggregate(x, axis, method):
    if method is None:
        return np.moveaxis(x, axis, 0)
    elif isinstance(method, np.ndarray):
        reference = method
        return np.sqrt(np.sum(x, axis=axis) / np.sum(reference, axis=axis))
    elif method == "mean":
        return np.nanmean(x, axis=axis)
    elif method == "max":
        return np.nanmax(x, axis=axis)
    elif method == "min":
        return np.nanmin(x,axis=axis)
    raise ValueError(f"Unknown aggregation method: {method}")


def _flatten_nested_dict(nested_dict: dict[tuple, Any]) -> dict:
    flat_dict = {}
    for k, v in nested_dict.items():
        if isinstance(k, tuple) and isinstance(v, dict):
            for k2, v2 in v.items():
                flat_dict[(*k, k2)] = v2
    return flat_dict


# -------------------------
# Extraction / Aggregation: Errors (with ddpm handling and flexible agg)
# -------------------------
def extract_val_result_vs_latent(
    results: Dict,
    res_type: Optional[str] = "errors", # One of "errors" or key to info dict
    methods: Optional[Iterable[str]] = None,
    aes: Optional[Iterable[str]] = None,
    regs: Optional[Iterable[bool]] = None,
    lat_dims: Optional[Iterable[int]] = None,
    ae_dyns: Optional[Iterable[Any]] = None,
    rep_idx: Optional[slice] = slice(None),
    traj_idx: Optional[slice] = slice(None),
    t_idx: Optional[slice] = slice(None),
    agg_reps: Optional[str] = "mean",
    agg_time: Optional[str] = "mean",
    agg_traj: Optional[str] = "mean",
    agg_order: tuple[int, int, int] = (0, 1, 2), # 0: repetitions, 1: traj, 2: time
) -> Dict[Tuple[str, str, bool], Dict[int, Dict[str, np.ndarray]]]:
    """
    Extract and aggregate errors vs latent dimension.

    Returns:
      mapping (method, ae, reg) -> { lat_dim -> {
            'n': int,
            'error_means': np.array(3,),
            'error_stds': np.array(3,),
            optional (ddpm):
              'error_min': np.array(3,),
              'error_max': np.array(3,)
        } }

    Parameters:
      - agg: one of 'mean', 'max', 'min' or a ndarray of same dimension as error.
      ndarray of same dimension is used as reference energy to compute relative errors.
    """
    for agg in [agg_reps, agg_time, agg_traj]:
        assert (isinstance(agg, np.ndarray) or agg in ("mean", "max", "min", None)), "agg must be one of 'mean', 'max', 'min', np.ndarray or None"
    axis_to_agg = {0: agg_reps, 1: agg_traj, 2: agg_time}
    n_aggs = sum(a is not None for a in axis_to_agg.values())

    out = {}
    for method, ae, reg, ae_dyn, lat_dim, value in _iter_selected_entries(results, methods, aes, regs, lat_dims, ae_dyns):
        if value is None or len(value) == 0:
            continue

        if res_type == "errors" or res_type == "error":
            result_tuple = value[0]
        else:
            result = value[1]
            if isinstance(result, list):
                # Repeats are saved as a list along the dicts
                result_tuple = (torch.stack([r[res_type] for r in result], dim=0),)
            else:
                result_tuple = (result[res_type],)

        key = (method, ae, reg, ae_dyn)
        out.setdefault(key, {})

        agg_central = []
        agg_std = []
        agg_min = []
        agg_max = []
        n = 1
        for e in [_to_numpy(t) for t in result_tuple]:
            if e.ndim == 2:
                e = np.expand_dims(e, 0)
            e = e[rep_idx, traj_idx, t_idx]
            e = np.moveaxis(e, tuple(reversed(agg_order)), (0, 1, 2))
            for i, agg_axis in enumerate(agg_order):
                if i == n_aggs - 1: # Compute variation on final aggregate
                    n = e.shape[-1]
                    if n > 2:
                        agg_std.append(np.nanstd(e, axis=-1))
                        agg_min.append(np.nanmin(e, axis=-1))
                        agg_max.append(np.nanmax(e, axis=-1))

                agg_method = axis_to_agg[agg_axis]
                e = _aggregate(e, -1, agg_method)
            agg_central.append(e)

        out[key][int(lat_dim)] = {
            "n": int(n),
            "agg_central": agg_central,
        }

        if n > 2:
            out[key][int(lat_dim)]["agg_stds"] = agg_std
            out[key][int(lat_dim)]["agg_min"] = agg_min
            out[key][int(lat_dim)]["agg_max"] = agg_max
    return out


# -------------------------
# Plotting: Errors vs latent dimension (supports ddpm min/max shading)
# -------------------------
def plot_errors_vs_latent(
    agg_data: Dict[Tuple[str, str, bool, str], Dict[int, Dict[str, np.ndarray]]],
    *,
    error_labels: Optional[List[str]] = None,
    colors: Optional[Dict[Tuple[str, str, bool, str], str]] = None,
    figsize: Tuple[int, int] = (16, 8),
    dpi: float = 100,
    show_std: bool = False,
    show_minmax: bool = False,
    title: Optional[str] = None,
    show_e_full: bool = True,
    show_e_svd: bool = True,
    show_e_ae: bool = True,
):
    """
    Plot three subplots (one per error type) vs latent dimension.

    If ddpm entries include 'error_min' and 'error_max', and show_minmax=True,
    a shaded band between min and max is drawn.
    """
    if error_labels is None:
        error_labels = ["error_full", "error_svd", "error_ae"]

    n_subplots = sum([show_e_full, show_e_svd, show_e_ae])
    fig, axes = plt.subplots(1, n_subplots, figsize=figsize, sharey=False, dpi=dpi)
    axes = [axes] if isinstance(axes, plt.Axes) else axes
    for ax_idx, ax in enumerate(axes):
        ax.set_xlabel("latent dimension")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_title(error_labels[ax_idx])
        ax.grid(True, linestyle=":", alpha=0.6)

    for key, latmap in agg_data.items():
        method, ae, reg, ae_dyn = key
        label = f"{method}/{ae}" if ae_dyn is None else f"{method}/{ae}/{ae_dyn}"
        lat_sorted = sorted(latmap.items(), key=lambda x: x[0])
        if not lat_sorted:
            continue
        lat_dims = np.array([ld for ld, _ in lat_sorted])
        error_agg = np.vstack([v["agg_central"] for _, v in lat_sorted])  # shape (n_lat, 3)

        has_minmaxsvd = all(("agg_min" in v and "agg_max" in v and "agg_stds" in v) for _, v in lat_sorted)
        if has_minmaxsvd:
            stds = np.vstack([v["agg_stds"] for _, v in lat_sorted])
            mins = np.vstack([v["agg_min"] for _, v in lat_sorted])
            maxs = np.vstack([v["agg_max"] for _, v in lat_sorted])
        else:
            mins = maxs = stds = None

        color = None
        if colors and key in colors:
            color = colors[key]

        i_e = [[0, 1, 2][show] for show in [show_e_full, show_e_svd, show_e_ae] if show]
        for i, ax in enumerate(axes):
            y = error_agg[:, i_e[i]]
            marker = {"ddpm": 'o', "1step": '*', "dlrom": '^'}[method]
            ax.plot(lat_dims, y, marker=marker, label=label, color=color)
            ax.set_ylabel('mean relative error')
            if show_std and has_minmaxsvd:
                yerr = stds[:, i_e[i]]
                ax.fill_between(lat_dims, y - yerr, y + yerr, alpha=0.2, color=color)
            if show_minmax and has_minmaxsvd:
                ax.fill_between(lat_dims, mins[:, i_e[i]], maxs[:, i_e[i]], alpha=0.12, color=color)

    for ax in axes:
        ax.legend(fontsize=12)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig

# -------------------------
# Plotting: time series
# -------------------------
def plot_time_series(
    time_data: Dict[Tuple[str, str, bool, str, int], Dict[str, np.ndarray]],
    time_axis: Optional[np.ndarray] = None,
    colors: Optional[Dict[Tuple[str, str, bool, str, int], str]] = None,
    show_std: bool = False,
    show_minmax: bool = False,
    scale = 1,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    ncol_legend: int = 1,
    *args,
    **kwargs
):
    """
    Plot data over time for selected entries.

    Parameters:
      - time_data: flattened output of extract_val_results_vs_latent
      - time_axis: optional 1D array of length ntimesteps
    """

    ax = ax or plt.gca()
    ax.axvspan(0, 10, color='0.9', label='_nolegend_')
    first = next(iter(time_data.values()))
    nt = first["agg_central"][0].shape[0]
    if time_axis is None:
        time_axis = np.arange(nt)
    else:
        time_axis = np.asarray(time_axis)
        assert time_axis.shape[0] == nt, "time_axis length mismatch"

    for key, v in time_data.items():
        method, ae, reg, ae_dyn, lat = key
        label = get_label(method, ae, lat, ae_dyn=ae_dyn)
        central = v["agg_central"][0] / scale

        color = None
        if colors and key in colors:
            color = colors[key]
        ax.plot(time_axis, central, label=label, color=color, *args, **kwargs)
        if show_std and "agg_stds" in v:
            std = v["agg_stds"][0] / scale
            ax.fill_between(time_axis, central - std, central + std, alpha=0.2, color=color, label='_nolegend_')
        if show_minmax and "agg_min" in v:
            agg_min = v["agg_min"][0] / scale
            agg_max = v["agg_max"][0] / scale
            ax.fill_between(time_axis, agg_min, agg_max, alpha=0.12, color=color, label='_nolegend_')
    ax.set_xlabel("time (s)", fontsize=12)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=12, ncol=ncol_legend)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=14)
    if title:
        ax.set_title(title)


def get_label(method, ae ,lat_dim, ae_dyn=None):
    if method == 'ddpm':
        method_str = 'DDP-ROM'
    elif method == 'dlrom':
        method_str = 'POD-DL-ROM'
    else:
        raise ValueError(f"Method {method} not supported")
    if ae == 'mlp':
        ae_str = '/mlp-AE'
    elif ae == 'conv':
        ae_str = '/conv-AE'
    elif ae == 'innae':
        ae_str = '/inv-AE'
    else:
        raise ValueError(f"AE {ae} not supported")
    ae_str = "" # Plot only one ae
    return rf"{method_str}{ae_str}/$N_{{\tilde{{y}}}}={lat_dim}$"


def add_kinetic_reference(time, traj_idx, time_steps, ax=None):
    ax = ax or plt.gca()
    ax.plot(time, test_set_ke['kinetic'][traj_idx, time_steps], label="ground truth", linestyle="--",
             color='r', linewidth=3)
    ax.legend()


def plot_score_vs_parameters(parameters, score, *args, **kwargs):
    plt.figure()
    plt.scatter(parameters[:, 0, 0], parameters[:, 0, 1], c=score, norm=colors.LogNorm(), *args, **kwargs)
    plt.xlabel("Angle of attack (rad)")
    plt.ylabel("Inflow velocity (m/s)")
    plt.colorbar(label="Mean relative error")


def rank_traj_difficulty(agg_error, ntraj=125):
    # errors are per trajectory
    error_cum = np.zeros((ntraj,))
    n_models = 0
    for k, v in agg_error.items():
        for lat_dim, error in v.items():
            error_cum += error['agg_central'][0]
            n_models += 1
    ranking = np.argsort(error_cum)
    return error_cum / n_models, ranking


def normalize_val_results(res, factor):
    if isinstance(res, dict):
        return {k: normalize_val_results(v, factor) for k, v in res.items()}
    elif isinstance(res, tuple):
        err = res[0]
        err_norm = (np.sqrt(err[0] / factor), *err[1:])
        tup_norm = (err_norm, *res[1:])
        return tup_norm
    return res


def format_result_table(agg):
    def format_float_tex(x):
        s = f"{x:.2e}"
        base, exp = s.split('e')
        exp = int(exp)  # remove leading zeros
        return f"{base} \\cdot 10^{{{exp}}}"

    def get_method_name(method):
        name = ''
        if method[0] == 'ddpm':
            name += 'DDP-ROM'
        elif method[0] == 'dlrom':
            name += 'POD DL-ROM'
        if method[1] == 'mlp':
            name += ' mlp-AE'
        elif method[1] == 'conv':
            name += ' conv-AE'
        elif method[1] == 'innae':
            name += ' inv-AE'
        return name
    # Collect all latent dimensions (columns)
    latent_dims = sorted({
        ld for method in agg for ld in agg[method]
    })

    # Start building LaTeX
    table = []
    table.append("\\begin{tabular}{l" + "c" * len(latent_dims) + "}")
    table.append("\\toprule")

    # Header row
    header = ["Method"] + [str(ld) for ld in latent_dims]
    table.append(" & ".join(header) + " \\\\")
    table.append("\\midrule")

    # Rows per method
    for method in agg:
        method_name = get_method_name(method)

        row_cells = [method_name]

        for ld in latent_dims:
            if ld not in agg[method]:
                row_cells.append("")  # empty cell
                continue

            entry = agg[method][ld]

            # Always have agg_central
            central = entry["agg_central"][0]
            central_str = format_float_tex(central)

            # If ddpm in method tuple, also include std
            # if any("ddpm" in str(m).lower() for m in method):
            #     std = entry["agg_stds"][0]
            #     std_str = f"{std:.2e}"
            #     cell = f"${central_str} \\pm {std_str}$"
            # else:
            cell = f"${central_str}$"

            row_cells.append(cell)

        table.append(" & ".join(row_cells) + " \\\\")

    table.append("\\bottomrule")
    table.append("\\end{tabular}")

    return "\n".join(table)



if __name__ == "__main__":
    mode = 'presentation' # 'presentation' adds custom changes to font sizes etc
    #mode = 'paper'

    val_results = load_rom_errors(filename="results/rom_val/rom_validation.pickle")
    energy_full = _to_numpy(val_results["energy_full"])
    val_results_normalized = normalize_val_results(val_results, energy_full)

    with open('results/rom_val/test_set_ke.pickle', 'rb') as f_new:
        test_set_ke = pickle.load(f_new)

    parameters = h5py.File("flowpastobject_2d/FlowAroundObstacle_data_20sec/full/test/FlowAroundObstacle_data_20sec.hdf5", "r")['MU']

    #lat_dims = [2, 3, 4, 5, 10, 20]
    lat_dims = [2, 5, 10]
    t_start = 0
    t_end = 400
    time_steps = slice(t_start,t_end)
    time_axis = np.arange(t_start, t_end) / 20

    # Find error against parameters
    agg = extract_val_result_vs_latent(val_results,
                                       methods=["dlrom","ddpm"],
                                       aes=['innae'],
                                       ae_dyns=None,
                                       lat_dims=lat_dims,
                                       t_idx=time_steps,
                                       agg_time=energy_full,
                                       agg_traj=None,
                                       agg_order = (2, 1, 0)
                                       )
    score, rank = rank_traj_difficulty(agg)
    plot_score_vs_parameters(parameters, score)
    plt.title("Errors ddpm/dlrom (all)")
    plt.show()

    print(f'Trajectory with lowest error: {rank[0]}: {score[rank[0]]}\t AoA/Inflow: {parameters[rank[0],0,0]}/{parameters[rank[0],0,1]}')
    print(f'Trajectory with highest error: {rank[-1]}: {score[rank[-1]]}\t AoA/Inflow: {parameters[rank[-1],0,0]}/{parameters[rank[-1],0,1]}')

    agg = extract_val_result_vs_latent(val_results,
                                       methods=["dlrom"],
                                       aes=['innae'],
                                       ae_dyns=None,
                                       lat_dims=lat_dims,
                                       t_idx=time_steps,
                                       agg_time=energy_full,
                                       agg_traj=None,
                                       agg_order=(2, 1, 0)
                                       )
    score_dlrom, rank_dlrom = rank_traj_difficulty(agg)
    agg = extract_val_result_vs_latent(val_results,
                                       methods=["ddpm"],
                                       aes=['innae'],
                                       ae_dyns=None,
                                       lat_dims=lat_dims,
                                       t_idx=time_steps,
                                       agg_time=energy_full,
                                       agg_traj=None,
                                       agg_order=(2, 1, 0)
                                       )
    score_ddpm, rank_ddpm = rank_traj_difficulty(agg)
    min_errors = np.minimum(score_dlrom, score_ddpm)
    max_errors = np.maximum(score_dlrom, score_ddpm)
    rank_min = np.argsort(min_errors)
    rank_max = np.argsort(max_errors)

    print(
        f'Trajectory with max min ave error: {rank_min[-1]}: {score[rank_min[-1]]}\t AoA/Inflow: {parameters[rank_min[-1], 0, 0]}/{parameters[rank_min[-1], 0, 1]}')
    print(
        f'Trajectory with min max ave error: {rank_max[0]}: {score[rank_max[0]]}\t AoA/Inflow: {parameters[rank_max[0], 0, 0]}/{parameters[rank_max[0], 0, 1]}')

    # Plot against latent space dimension
    agg = extract_val_result_vs_latent(val_results, methods=['dlrom','ddpm'], aes=None, ae_dyns=[None],
                                       lat_dims=lat_dims, agg_time=energy_full[:, time_steps], t_idx=time_steps, agg_order=(2, 1, 0),)
    tab = format_result_table(agg)
    print(tab)
    fig = plot_errors_vs_latent(agg, title=None,
                                show_minmax=False,
                                show_std=False,
                                show_e_full=True,
                                show_e_svd=False,
                                show_e_ae=False,
                                error_labels=[None,None,None],
                                figsize=(7,7),
                                dpi=200)
    axs = fig.get_axes()
    for ax in axs:
        ax.set_yscale('log')
    fig.tight_layout()
    fig.show()

    # Plot over time

    # Single trajectories
    fig_k = plt.figure(figsize=(15, 10))
    fig_e = plt.figure(figsize=(15, 10))
    ax1, ax2, ax3, ax4 = None, None, None, None
    most_similar_result = [rank_min[-1], rank_max[0]]
    for i, traj_idx in enumerate([28, 46]):
        ax1 = fig_k.add_subplot(2, 2, 1+2*i)
        agg = extract_val_result_vs_latent(val_results, methods=["ddpm"], aes=["innae"], ae_dyns=[None],
                                           lat_dims=lat_dims,
                                           traj_idx=slice(traj_idx,traj_idx+1),
                                           t_idx=time_steps,
                                           res_type="kinetic",
                                           agg_order=(1, 0, 2), agg_time=None)
        plot_time_series(_flatten_nested_dict(agg), title=fr"$\alpha_{{in}}$: {parameters[traj_idx,0,0]:.2f} $\gamma_{{in}}$: {parameters[traj_idx,0,1]:.2f}",
                            time_axis=time_axis,
                            ylabel="kinetic energy",
                            ax=ax1)
        add_kinetic_reference(time_axis, traj_idx, time_steps, ax=ax1)
        ax1.set_ylim((0,torch.max(test_set_ke['kinetic'][traj_idx, time_steps])*1.25))

        ax2 = fig_k.add_subplot(2, 2, 2+2*i, sharey=ax1, sharex=ax1)
        agg = extract_val_result_vs_latent(val_results, methods=["dlrom"], aes=["innae"], ae_dyns=[None],
                                           lat_dims=lat_dims,
                                           traj_idx=slice(traj_idx,traj_idx+1),
                                           t_idx=time_steps,
                                           res_type="kinetic",
                                           agg_order=(1, 0, 2), agg_time=None, agg_traj=None)
        plot_time_series(_flatten_nested_dict(agg), title=fr"$\alpha_{{in}}$: {parameters[traj_idx,0,0]:.2f} $\gamma_{{in}}$: {parameters[traj_idx,0,1]:.2f}",
                            time_axis=time_axis,
                            ylabel="kinetic energy",
                            ax=ax2)
        add_kinetic_reference(time_axis, traj_idx, time_steps, ax=ax2)
        ax2.set_ylim((0, torch.max(test_set_ke['kinetic'][traj_idx, time_steps]) * 1.3))

        ax3 = fig_e.add_subplot(2, 2, 1+2*i, sharey=ax3)
        agg = extract_val_result_vs_latent(val_results_normalized, methods=["ddpm"], aes=["innae"], ae_dyns=[None],
                                           lat_dims=lat_dims,
                                           traj_idx=slice(traj_idx,traj_idx+1), res_type="error",
                                           agg_order=(1, 0, 2), agg_time=None)
        plot_time_series(_flatten_nested_dict(agg), title=fr"$\alpha_{{in}}$: {parameters[traj_idx,0,0]:.2f} $\gamma_{{in}}$: {parameters[traj_idx,0,1]:.2f}",
                        ylabel="prediction error",
                        time_axis=time_axis,
                        ax=ax3)

        ax4 = fig_e.add_subplot(2, 2, 2+2*i, sharex=ax3, sharey=ax3)
        agg = extract_val_result_vs_latent(val_results_normalized, methods=["dlrom"], aes=["innae"], ae_dyns=[None], lat_dims=lat_dims,
                                           traj_idx=slice(traj_idx,traj_idx+1), res_type="error",
                                           agg_order=(1, 0, 2), agg_time=None)
        plot_time_series(_flatten_nested_dict(agg), title=fr"$\alpha_{{in}}$: {parameters[traj_idx,0,0]:.2f} $\gamma_{{in}}$: {parameters[traj_idx,0,1]:.2f}",
                         ylabel="prediction error",
                         time_axis=time_axis,
                         ax=ax4)
    ax3.set_ylim(0, 0.75)

    if mode == 'presentation':
        for ax in fig_k.axes:
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.get_legend().remove()
            ax.xaxis.label.set_size(16)
            ax.yaxis.label.set_size(16)
            ax.set_ylim((120,275))
            ax.set_box_aspect(0.4)
        fig_k.axes[0].legend([f'dim={dim}' for dim in lat_dims], bbox_to_anchor=(0.99, 0.05),
                             loc='lower right', fontsize=15)
        # fig_k.axes[2].legend([f'dim={dim}' for dim in lat_dims], bbox_to_anchor=(0.99, 0.05),
        #                      loc='lower right', fontsize=15)
        fig_k.axes[3].legend([f'dim={dim}' for dim in lat_dims], bbox_to_anchor=(0.99, 0.05),
                             loc='lower right', fontsize=15)
        fig_k.axes[2].set_title('DDP-ROM',fontsize=17)
        fig_k.axes[2].xaxis.set_visible(False)
        fig_k.axes[3].set_title('POD-DL-ROM',fontsize=17)
        fig_k.axes[1].set_ylabel('')
    fig_k.tight_layout()
    fig_e.tight_layout()
    fig_k.show()
    fig_e.show()

    # Error over time averaged over trajectories in test set

    # Compare all ddpms
    fig, axs = plt.subplots(3,3, figsize=(18,18),sharex=True, sharey=True)
    for i, backbone in enumerate(['innae','mlp','conv']):
        for j, ae_d in enumerate([None, '1step', 'dlrom']):
            break
            agg = extract_val_result_vs_latent(val_results_normalized, methods=["ddpm"], aes=[backbone], ae_dyns=[ae_d], lat_dims=lat_dims,
                                               res_type="error",
                                               agg_order=(1, 0, 2), agg_time=None)
            plot_time_series(_flatten_nested_dict(agg), title="Average error over time",
                             ylabel="relative error",
                             time_axis=time_axis,
                             show_std=True,
                             ax=axs[i,j])
            axs[i,j].set_yscale('log')
    fig.tight_layout()
    fig.show()

    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True, sharey=True)
    for i, method in enumerate(['ddpm','dlrom']):
        agg = extract_val_result_vs_latent(val_results_normalized, methods=[method], aes=["innae"], ae_dyns=[None], lat_dims=lat_dims,
                                           res_type="error",
                                           agg_order=(1, 0, 2), agg_time=None)
        plot_time_series(_flatten_nested_dict(agg),
                         ylabel="prediction error",
                         time_axis=time_axis,
                         ax=axs[i],
                         ncol_legend=1)
        axs[i].set_yscale('log')

    if mode == 'presentation':
        axs[0].set_title('DDP-ROM',fontsize=17)
        axs[0].xaxis.set_visible(False)
        axs[1].set_title('POD-DL-ROM',fontsize=17)
        fig.text(0.01, 0.5, 'prediction error', va='center', rotation='vertical', fontsize=16)
        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.get_legend().remove()
            ax.set_ylim([5e-3, 2])
            ax.set_ylabel('')
            ax.xaxis.label.set_size('x-large')
            ax.set_box_aspect(0.4)
        fig.legend([f'dim={dim}' for dim in lat_dims], bbox_to_anchor=(0.9,0.15), loc='lower right', fontsize='large')
    fig.tight_layout()
    fig.show()

    ## DDPM variability

    traj_idx = [11, 28]
    fig_var = plt.figure(figsize=(15, 6))
    ax1 = fig_var.add_subplot(1, 2, 1)
    agg = extract_val_result_vs_latent(val_results_normalized, methods=["ddpm"], aes=["innae"], ae_dyns=[None],
                                       lat_dims=[5],
                                       traj_idx=slice(traj_idx[1],traj_idx[1]+1), res_type="error",
                                       agg_order=(1, 0, 2), agg_time=None)
    plot_time_series(_flatten_nested_dict(agg), title=fr"$\alpha_{{in}}$: {parameters[traj_idx[1],0,0]:.2f} $\gamma_{{in}}$: {parameters[traj_idx[1],0,1]:.2f}",
                    ylabel="prediction error",
                    time_axis=time_axis,
                    show_minmax=True,
                    ax=ax1)

    ax2 = fig_var.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    agg = extract_val_result_vs_latent(val_results_normalized, methods=["ddpm"], aes=["innae"], ae_dyns=[None],
                                       lat_dims=[2],
                                       traj_idx=slice(traj_idx[0],traj_idx[0]+1), res_type="error",
                                       agg_order=(1, 0, 2), agg_time=None)
    plot_time_series(_flatten_nested_dict(agg), title=fr"$\alpha_{{in}}$: {parameters[traj_idx[0],0,0]:.2f} $\gamma_{{in}}$: {parameters[traj_idx[0],0,1]:.2f}",
                     ylabel="prediction error",
                     time_axis=time_axis,
                     show_minmax=True,
                     ax=ax2)
    fig_var.tight_layout()
    fig_var.show()
