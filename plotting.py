import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    import seaborn as sns
except ImportError:
    pass
import sys

import numpy as np
from scipy.fft import fftshift
from lambda_loop import set_diff2d

from lib import Residual
from single_step_newton import get_minimum_demerit_resid
from scipy.fft import fftshift, fftfreq
from densify import get_dense_solution

import logger

log = logger.get_logger(__name__)


def autoscale_y(ax, margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        log.info(f"FOO ")
        y_displayed = yd[((xd > lo) & (xd < hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed) - margin * h
        top = np.max(y_displayed) + margin * h
        return bot, top

    lines = ax.get_lines()
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot:
            bot = new_bot
        if new_top > top:
            top = new_top

    ax.set_ylim(bot, top)


def inspect_wavefield_vs_model(
    wavefield,
    io,
    step,
    setup={
        "density": True,
        "log": True,
    },
    threshold_factor=0.25,
):
    """This function plots the wavefield and the model."""
    setup["bins"] = 100

    n, bins_full, _ = plt.hist(
        np.log10(np.abs(wavefield[wavefield.astype(bool)]).flatten()),
        **setup,
        facecolor="blue",
        label="noisy wavefield",
    )
    setup["bins"] = bins_full

    data = np.log10(np.abs(io["models"][step][io["models"][step].astype(bool)].flatten()))
    if np.max(data) > np.max(bins_full):
        count = 0
        increment = np.diff(bins_full)[0]
        while np.max(data) > np.max(bins_full):
            bins_full = np.append(bins_full, bins_full[-1] + increment)
            count += 1
        setup["bins"] = bins_full
        log.info(f"Added {count} bins, max {np.max(data):.2f}")

    if np.min(data) < np.min(bins_full):
        warn = "- min range fail"
        log.warning("min range exceeded")
    else:
        warn = ""

    _ = plt.hist(
        data,
        **setup,
        # edgecolor="orange",
        # facecolor="orange",
        label=f"wavefield after λ step {warn}",
    )

    data = np.log10(np.abs(io["models_λ"][step][io["models_λ"][step].astype(bool)].flatten()))
    if np.max(data) > np.max(bins_full):
        count = 0
        increment = np.diff(bins_full)[0]
        while np.max(data) > np.max(bins_full):
            bins_full = np.append(bins_full, bins_full[-1] + increment)
            count += 1
        setup["bins"] = bins_full
        log.info(f"Added {count} bins, max {np.max(data):.2f}")

    if np.min(data) < np.min(bins_full):
        warn = "- min range fail"
        log.warning("min range exceeded")
    else:
        warn = ""

    _ = plt.hist(
        data,
        **setup,
        edgecolor="green",
        facecolor="none",
        label=f"wavefield before first ht {warn}",
    )

    threshold = threshold_factor * io["lambdas"][step] / io["Ls"][step]

    _ = plt.axvline(np.log10(threshold), color="r")
    _ = plt.legend()
    _ = plt.title(f"step {step}")

    log.info(f"threshold: {threshold} = 10^{np.log10(threshold):.2f}")
    log.info(f"smallest: {np.power(10, np.min(data))} = 10^{np.min(data):.2f}")


def make_plots(
    io,
    step,
    xrange=None,
    yrange=None,
    x_centre=None,
    y_centre=None,
    init_coords=[0, 0],
    delay_axis=1,
    cmap="cubehelix",
):
    if xrange is None:
        xrange = int(io["models"][step].shape[1 - delay_axis] / 2)
    if x_centre is None:
        x_centre = int(io["models"][step].shape[1 - delay_axis] / 2)

    if yrange is None:
        yrange = int(io["models"][step].shape[delay_axis] / 2)
    if y_centre is None:
        y_centre = int(io["models"][step].shape[delay_axis] / 2)

    if "models_λ" in io.keys():
        if step in io["models_λ"].keys():
            plt.figure()
            coords_λ = np.nonzero(fftshift(io["models_λ"][step]))
            plt.scatter(coords_λ[0], coords_λ[1], s=[1] * len(coords_λ[0]))
            plt.title(f"model after λ>0 in step {step} with {np.count_nonzero(io['models_λ'][step])} components")
            plt.xlim(x_centre - xrange, x_centre + xrange)
            plt.ylim([y_centre - 12, y_centre + yrange])

            ax = plt.gca()
            ax.axhline(y_centre, color="r")
            ax.axvline(x_centre, color="r")

            ax.axhline(y_centre + init_coords[1], color="orange")
            ax.axvline(x_centre + init_coords[0], color="orange")

    if "models" in io.keys():
        if step in io["models"].keys():
            plt.figure()
            coords = np.nonzero(fftshift(io["models"][step]))
            plt.scatter(coords[0], coords[1], s=[1] * len(coords[0]))
            plt.title(f"model after λ=0 in step {step} with {np.count_nonzero(io['models'][step])} components")
            plt.xlim(x_centre - xrange, x_centre + xrange)
            plt.ylim([y_centre - 12, y_centre + yrange])

            ax = plt.gca()
            ax.axhline(y_centre, color="r")
            ax.axvline(x_centre, color="r")

            ax.axhline(y_centre + init_coords[1], color="orange")
            ax.axvline(x_centre + init_coords[0], color="orange")

            plt.figure()
            new_components = set_diff2d(
                np.transpose(np.nonzero(fftshift(io["models"][step]))),
                np.transpose(np.nonzero(fftshift(io["models"][step - 1]))),
            )
            new_components_scat = np.transpose(new_components.tolist())
            if len(new_components_scat) > 0:
                plt.scatter(new_components_scat[0], new_components_scat[1], s=[1] * len(new_components_scat[0]))
                plt.title(f"new components ({len(new_components_scat[0])}) in step {step} ")
                plt.xlim(x_centre - xrange, x_centre + xrange)
                plt.ylim([y_centre - 12, y_centre + yrange])

                ax = plt.gca()
                ax.axhline(y_centre, color="r")
                ax.axvline(x_centre, color="r")

                ax.axhline(y_centre + init_coords[1], color="orange")
                ax.axvline(x_centre + init_coords[0], color="orange")
            else:
                print(f"No new components in step {step}")

            plt.figure()
            plt.title(f"model with ({np.count_nonzero(io['models'][step])}) in step {step} ")
            plt.imshow(np.abs(fftshift(io["models"][step]).T), norm=LogNorm(), interpolation="none", cmap=cmap)
            plt.xlim(x_centre - xrange, x_centre + xrange)
            plt.ylim([y_centre - 12, y_centre + yrange])


# seaborn style
def set_style():
    sns.set_context("paper")
    sns.set_style(
        "white",
        {
            "font.family": "serif",
            "font.serif": ["Times", "Palatino", "serif"],
        },
    )


def set_rc(fontSize=None):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["image.origin"] = "lower"
    plt.rcParams["image.aspect"] = "auto"
    plt.rcParams["image.interpolation"] = "none"
    if fontSize is not None:
        plt.rcParams["font.size"] = str(fontSize)


def get_diag_plot(io, ax=None, max_step=None, tick_step=2, include_lambda_index=True):
    if ax is None:
        ax = plt.figure().gca()
    lambdas = {}
    all_n_comp = np.array([])
    all_demerits = np.array([])
    all_n_comp_zero = np.array([])

    lambdas["sub4_ls2"] = np.array([])

    lambda_loop_locators = np.array([0])
    lambda_loop_minor_locators = np.array([])

    if max_step is not None:
        iterator = range(1, max_step + 1)
    else:
        iterator = sorted(io["n_comp"].keys())

    for i in iterator:
        subloops = io["FISTAs"][i]
        all_demerits = np.append(all_demerits, io["demerits"][i])
        all_n_comp = np.append(all_n_comp, io["n_comp"][i])
        all_n_comp_zero = np.append(all_n_comp_zero, io["n_comp_zero"][i])
        lambdas["sub4_ls2"] = np.append(lambdas["sub4_ls2"], [io["lambdas"][i]] * io["niters"][i] * subloops)
        lambda_loop_locators = np.append(lambda_loop_locators, lambda_loop_locators[-1] + io["niters"][i] * subloops)
        for j in range(1, subloops):
            lambda_loop_minor_locators = np.append(
                lambda_loop_minor_locators, lambda_loop_locators[i - 1] + io["niters"][i] * j
            )

    start_at = 1
    span = range(start_at, len(all_demerits) + start_at)

    ax.plot(np.log10(all_demerits))

    ax.set_xticks(lambda_loop_locators, minor=False)
    ax.set_xticks(lambda_loop_minor_locators, minor=True)
    ax.set_xlabel("FISTA iterations", fontsize=14)
    ax.set_ylabel("log10(demerit)")

    # rotate all xtick labels
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    # and hide every second one or show every third:
    if tick_step == 2:
        for label in ax.xaxis.get_ticklabels()[1::tick_step]:
            label.set_visible(False)
    elif tick_step > 2:
        for i, label in enumerate(ax.xaxis.get_ticklabels()):
            if i % tick_step != 0:
                label.set_visible(False)

    # setup a secondary x axis:
    if include_lambda_index:
        tick_locs = ax.xaxis.get_ticklocs()

        labels = []
        for i, loc in enumerate(tick_locs):
            if i % tick_step == 0:
                # labels.append(rf"$\lambda_{{{i}}}$")
                labels.append(f"{i+1}")
            else:
                labels.append("")

        x_ax2 = ax.secondary_xaxis("top")
        x_ax2.tick_params(direction="in")
        x_ax2.set_xticks(tick_locs, labels=labels)
        x_ax2.set_xlabel(r"$\lambda$ iterations", fontsize=14)

    # setup a twin y axis:

    y_ax2 = ax.twinx()
    y_ax2.plot(all_n_comp, color="green", label="All")
    y_ax2.plot(all_n_comp_zero, color="orange", label="Approved")

    y_ax2.set_ylabel("Non-zero components")

    ax.xaxis.grid(linewidth=2)
    ax.xaxis.grid(linewidth=1, linestyle="dotted", which="minor")


def get_data_plot(data, ax, transpose=True, cmap="gray_r", cfreq=0, bw=0, subint_time=0, vmin_cdf_threshold=0.03):
    n, b = np.histogram(data.flatten())

    extra_opts = {}
    if vmin_cdf_threshold > 0:
        extra_opts["vmin"] = get_vmin_from_cdf(n, b, threshold=vmin_cdf_threshold)

    ax.set_aspect("auto")
    if transpose:
        time_axis = 0
    else:
        time_axis = 1
    if cfreq > 0 and bw > 0 and subint_time > 0:
        t_extra_str = " [mins]"
        f_extra_str = " [MHz]"
        extent = (0, subint_time * data.shape[time_axis] / 60, cfreq - bw / 2, cfreq + bw / 2)
    else:
        extent = None
        t_extra_str = " [arbitrary]"
        f_extra_str = " [arbitrary]"
    ax.imshow(data.T if transpose else data, cmap=cmap, rasterized=True, extent=extent, **extra_opts)
    ax.set_xlabel(f"Time{t_extra_str}")
    ax.set_ylabel(f"Frequency{f_extra_str}")


def get_wavefield_mesh(nchan: int, nsubint: int, bw=1.0, subint_time=8.0, flip_doppler=False):
    """Generate a wavefield for wavefield plots

    If bandwidth or subintegration time are zero, use arbitrary values

    Args:
        nchan (int): Number of channels in the corresponding dynamic spectrum.
        nsubint (int): Number of subintegrations in the corresponding dynamic spectrum.
        bw (float): Bandwidth in the corresponding dynamic spectrum. Defaults to 1 MHz
        subint_time (float): Time in every subintegration. Defaults to 8 s
        flip_doppler (bool, optional): . Defaults to False.

    Returns:
        tuple: tuple containing the x and y coordinates as generated by numpy.meshgrid
    """
    X, Y = np.meshgrid(fftshift(fftfreq(nsubint, subint_time)) * 1e3, fftshift(fftfreq(nchan, bw / nchan)))
    if flip_doppler:
        X = -X
    return X, Y


def get_vmin_from_cdf(hist, bins, threshold=0.03):
    cdf = hist.cumsum()
    cdf = cdf / cdf.max()

    cdf_index = np.argmin(np.abs(cdf - threshold))
    return (bins[cdf_index] + bins[cdf_index + 1]) / 2


def get_wavefield_plot(
    wavefield,
    ax,
    cmap="gray_r",
    bw=0,
    subint_time=0,
    flip_doppler=False,
    vmin_cdf_threshold=0.03,
):
    wf_log10_abs = np.log10(np.power(np.abs(fftshift(wavefield)), 2))
    extra_opts = {}
    if vmin_cdf_threshold > 0:
        n, b = np.histogram(wf_log10_abs[np.where(wf_log10_abs > -np.inf)])
        extra_opts["vmin"] = get_vmin_from_cdf(n, b, threshold=vmin_cdf_threshold)

    ds_label = "Doppler shift [mHz]"
    delay_label = r"Delay [$\mu$s]"
    if bw <= 0:
        ds_label = "Doppler shift [arbitrary]"
        bw = 1.0
    if subint_time <= 0:
        ds_label = "Delay [aribtrary]"
        subint_time = 8.0

    ax.set_aspect("auto")
    x, y = get_wavefield_mesh(wavefield.shape[0], wavefield.shape[1], bw, subint_time, flip_doppler=flip_doppler)  # type: ignore
    ax.pcolormesh(
        x,
        y,
        wf_log10_abs,
        cmap=cmap,
        rasterized=True,
        shading="auto",
        **extra_opts,
    )
    ax.set_xlabel(ds_label)
    ax.set_ylabel(delay_label)


def get_dense_wavefield_plot_from_sparse(
    io,
    data,
    chosen_step,
    ax,
    cmap="gray_r",
    bw=0,
    subint_time=0,
    flip_doppler=False,
    vmin_cdf_threshold=0.03,
    method="FISTA",
    **kwargs,
):
    dense_wavefield = get_dense_solution(
        io["models"][chosen_step], data, io["masks"][chosen_step], method=method, **kwargs
    )
    get_wavefield_plot(
        dense_wavefield,
        ax,
        cmap=cmap,
        bw=bw,
        subint_time=subint_time,
        flip_doppler=flip_doppler,
        vmin_cdf_threshold=vmin_cdf_threshold,
    )
    return None


def get_figsize(columnwidth=244.0, scale=3.5, panels=3):
    fig_width_pt = columnwidth * scale  # scale * output of \the\columwidth in latex
    dpi = 72.27 * scale / 2
    inches_per_pt = 1.0 / dpi
    golden_ratio = (np.sqrt(5) + 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width / golden_ratio * panels
    return (fig_width, fig_height)


def get_paper_figure(
    io,
    data,
    chosen_step,
    max_diagnostic_step=None,
    resid=None,
    outfn=None,
    scale=3.5,
    columnwidth=244.0,
    panels=3,
    cfreq=0,
    bw=0,
    subint_time=0,
    flip_doppler=False,
    dense=True,
    tick_step=2,
    vmin=0.03,
    vmin_data=0.03,
    fontSize=None,
    facecolor="white",
):
    if "seaborn" in sys.modules.keys():
        sns.set_theme()
        set_style()
    set_rc(fontSize)
    if max_diagnostic_step is None:
        max_diagnostic_step = chosen_step
    dpi = 72.27 * scale / 2
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=get_figsize(columnwidth, scale, panels), dpi=dpi, facecolor=facecolor)
    get_data_plot(data, axs[0], cfreq=cfreq, bw=bw, subint_time=subint_time, vmin_cdf_threshold=vmin_data)
    get_diag_plot(io, axs[1], tick_step=tick_step, max_step=max_diagnostic_step)
    if dense:
        get_wavefield_plot(
            resid.wavefield.T,
            axs[2],
            bw=bw,
            subint_time=subint_time,
            flip_doppler=flip_doppler,
            vmin_cdf_threshold=vmin,
        )
    else:
        get_wavefield_plot(
            io["models"][chosen_step].T,
            axs[2],
            bw=bw,
            subint_time=subint_time,
            vmin_cdf_threshold=vmin,
            flip_doppler=flip_doppler,
        )
    fig.tight_layout()
    fig.align_ylabels(axs[:])
    if outfn is not None:
        fig.savefig(outfn)
    return fig, axs


def get_dynamic_field_plot(data, io, step, cmap="cubehelix", figsize=(16, 8)):
    resid = Residual(data, io["models"][step], None, io["masks"][step])
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=figsize)
    _ = axs[0].imshow(np.abs(resid.H.T), cmap=cmap)
    _ = axs[0].set_title("magnitudes")
    _ = axs[0].set_ylabel("Frequency [arbitrary]")
    _ = axs[0].set_xlabel("Time [arbitrary]")
    _ = axs[1].set_xlabel("Time [arbitrary]")
    ims = axs[1].imshow(np.angle(resid.H.T), cmap=cmap)
    cax = axs[1].inset_axes([1.04, 0.2, 0.05, 0.6])
    fig.colorbar(ims, ax=axs[1], cax=cax, label="phase [rad]")
    _ = axs[1].set_title("phases")
    fig.suptitle(f"Dynamic field at step {step}")
    return fig, axs
