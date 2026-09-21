"""
Figure-making for the pipeline's diagnostic plots.

The per-night, corner and residual-histogram figures here are the same ones
that go into the paper: the drawing code was lifted from the paper-figure
notebooks and scripts in `__make_paper_figs/` so that a stage-3 run and a
paper figure can't drift apart. Their call signatures are unchanged, so
`postprocess.py` drives them exactly as before -- only what comes out of them
is different.
"""

import os

import arviz as az
import corner
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.time import Time
from matplotlib.colors import Normalize
from scipy.stats import norm, skewcauchy

import mixture_model

# The style every paper figure was made in. The '-white' variant differs from
# plain 'seaborn-v0_8' only in dropping the grey axes background and grid,
# which is what lets the light grey posterior bands below read as posterior
# bands rather than as part of the axes.
plt.style.use('seaborn-v0_8-white')

# ----------------------------
# Shared style: RFI colour code
# ----------------------------
# Every per-sample marker in every panel is coloured by which flagger called it
# RFI, so a sample can be followed by eye from the data panel down through the
# residual and soft-label panels.
C_MODEL = "#04D9FF"
C_SSINS = "lime"
C_BOTH  = "red"

# Drawn (and listed in the legend) in this order -- don't reorder. `s` is a
# scatter marker size: points SQUARED, i.e. an area. Only the square and the
# diamond are outlined; the star is drawn bare.
RFI_STYLES = {
    "model_only": dict(label="RFI: model only", marker="s", color=C_MODEL,
                       s=18, zorder=3, edge=True),
    "ssins_only": dict(label="RFI: SSINS only", marker="D", color=C_SSINS,
                       s=12, zorder=4, edge=True),
    "both":       dict(label="RFI: both",       marker="*", color=C_BOTH,
                       s=18, zorder=5, edge=False),
}

# Widths of the shaded/errorbar posterior intervals, in posterior standard
# deviations. The background band is drawn wider than the residual and
# soft-label error bars purely for legibility at the scale of the data panel.
N_SIGMA_BG        = 10
N_SIGMA_ERRORBARS = 5

# A star encloses much less ink than a square or diamond of the same nominal
# area -- its points are thin -- so it reads small wherever it isn't sitting
# out on an open data curve. The residual and soft-label panels, where it has
# to be picked out of a dense band of grey/black dots, size it up; the data
# panel keeps the smaller size.
ERRORBAR_STAR_SIZE = 45


def markersize(s):
    """
    Convert a scatter marker size to the equivalent `Line2D` marker size.

    The two APIs used across these panels do NOT take the same units:
    `Axes.scatter` sizes markers by AREA in points squared, while the markers
    `Axes.errorbar` draws are `Line2D` markers sized by DIAMETER in points.
    Both scale the same unit marker path, so matching them exactly is just
    sqrt -- without it a category's markers come out visibly bigger in the
    residual/soft-label panels than in the data panel.
    """
    return np.sqrt(s)


def rfi_masks(model_mask, ssins_mask):
    """
    Split one night's flagged samples into the three RFI categories the
    overlays are colour-coded by.

    Parameters
    ----------
    model_mask : ndarray of bool, shape (T,)
        True where the model's hard label says RFI.
    ssins_mask : ndarray of bool, shape (T,)
        True where SSINS' match filter flagged the sample.

    Returns
    -------
    dict
        Category name -> boolean mask, keyed to match `RFI_STYLES`. The masks
        are disjoint; samples in none of them are clean by both measures and
        are left to the panel's base series.
    """
    model_mask = np.asarray(model_mask).astype(bool)
    ssins_mask = np.asarray(ssins_mask).astype(bool)

    return {
        "model_only":  model_mask & ~ssins_mask,
        "ssins_only": ~model_mask &  ssins_mask,
        "both":        model_mask &  ssins_mask,
    }


def _styles(star_size=None):
    """`RFI_STYLES`, optionally with the star resized for a given figure."""
    if star_size is None:
        return RFI_STYLES

    return {name: (dict(style, s=star_size) if name == "both" else style)
            for name, style in RFI_STYLES.items()}


def scatter_rfi(ax, x, y, masks, star_size=None, label=True):
    """Overlay the flagged samples on a data panel, one call per category."""
    for name, style in _styles(star_size).items():
        mask = masks[name]
        ax.scatter(
            x[mask], y[mask],
            marker=style['marker'],
            s=style['s'],
            color=style['color'],
            zorder=style['zorder'],
            label=style['label'] if label else None,
            **(dict(edgecolors="black", linewidths=0.4) if style['edge'] else {}),
        )


def errorbar_rfi(ax, x, y, yerr, masks, star_size=None):
    """
    Overlay the flagged samples on an errorbar panel, one call per category.

    Unlabelled on purpose: the colour code is already spelled out in the
    legend of the data panel above.
    """
    for name, style in _styles(star_size).items():
        mask = masks[name]
        ax.errorbar(
            x[mask], y[mask], yerr=yerr[mask],
            fmt=style['marker'],
            markersize=markersize(style['s']),
            color=style['color'],
            ecolor=style['color'],
            elinewidth=0.8,
            zorder=style['zorder'],
            **(dict(markeredgecolor="black", markeredgewidth=0.4)
               if style['edge'] else {}),
        )


def _moments(arr, ndx, stop):
    """
    Posterior mean and std over draws of one night's slice of an (S, N, T)
    array, trimmed to its `stop` valid timesteps and reduced in float64.

    The float64 cast is the point of this helper. The chains are float32, and
    a float32 reduction over ~144,000 draws of a quantity as large as the
    background (~780) relative to its posterior spread (~0.06) is dominated by
    accumulation error -- widen first, reduce second.
    """
    arr = np.asarray(arr[:, ndx, :stop], dtype=np.float64)

    return arr.mean(axis=0), arr.std(axis=0)


def plot_convergence_metrics(idata_summary, savefig=False, savepath=None, show=False):
    """
    """
    plt.figure(figsize=(6,3))
    plt.hist(np.array(idata_summary['r_hat'],dtype=float),bins=50)
    plt.title("Histogram of r-hat values for all sampled parameters, across all chains")
    plt.xticks(rotation=45)
    plt.yscale("log")
    plt.tight_layout()
    if savefig:
        plt.savefig(os.path.join(savepath,"rhat.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(6,3))
    plt.hist(np.array(idata_summary['ess_bulk'],dtype=float),bins=50)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title("Histogram of ess_bulk values for all sampled parameters")
    if savefig:
        plt.savefig(os.path.join(savepath,"ess.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_trace_corner(idata, savefig=False, savepath=None, show=False):
    """
    Trace plots and a corner plot of the three global mixture parameters.

    The corner plot is the paper's (`__make_paper_figs/make_corner_plots.ipynb`):
    it shows w_rfi rather than its complement w_clean -- the RFI occupancy is
    the quantity of interest, and it's what every other figure and table
    quotes -- with the parameters labelled as they are in the paper.
    """
    az.plot_trace(idata, var_names=[
                    'loc_rfi', 
                    'scale_rfi', 
                    'w_clean',
                ],
              figure_kwargs={"figsize": (12, 3)})
    plt.tight_layout()
    if savefig:
        plt.savefig(os.path.join(savepath,"trace_plot.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    posterior = az.extract(idata,
        var_names=["loc_rfi", "scale_rfi", "w_rfi"]
    ).to_dataframe()

    fig = plt.figure(figsize=(5, 5))

    corner.corner(
        posterior[["loc_rfi", "scale_rfi", "w_rfi"]].values,
        labels=[r"$\mu_\mathrm{RFI}$", r"$\sigma_\mathrm{RFI}$", r"$w_\mathrm{RFI}$"],
        show_titles=True,
        fig=fig,
        label_kwargs={"fontsize": 14},
        title_kwargs={"fontsize": 12},
        labelpad=0.1,
    )

    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=11)

    if savefig:
        plt.savefig(os.path.join(savepath,"param_corner.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_corner_mu_sigma(idata, savefig=False, savepath=None, show=False):
    """
    """
    mu_beta_samples = (
        idata.posterior['mu_beta']
        .stack(sample=('chain', 'draw'))
        .transpose('sample', ...)
        .values
    )

    degree = mu_beta_samples.shape[-1]
    labels = [f"mu_beta[{i}]" for i in range(degree)]

    corner.corner(
        mu_beta_samples,
        labels=labels,
        show_titles=True
    )
    plt.suptitle("mu_beta for each L mode", fontsize=60)
    if savefig:
        plt.savefig(os.path.join(savepath,"mu_beta_corner.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    sigma_beta_samples = (
        idata.posterior['sigma_beta']
        .stack(sample=('chain', 'draw'))
        .transpose('sample', ...)
        .values
    )

    degree = sigma_beta_samples.shape[-1]
    labels = [f"sigma_beta[{i}]" for i in range(degree)]

    corner.corner(
        sigma_beta_samples,
        labels=labels,
        show_titles=True
    )
    plt.suptitle("sigma_beta for each L mode", fontsize=60)
    if savefig:
        plt.savefig(os.path.join(savepath,"sigma_beta_corner.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def night_dates(night_ids):
    """
    UTC datetimes for a list of night ids.

    A night id is an MWA obsid with its last four digits dropped, and an obsid
    is its GPS start second, so `night * 10_000` is the GPS second the night
    starts at.
    """
    return np.array([Time(int(n) * 10_000, format='gps').utc.datetime
                     for n in night_ids])


def _posterior_mean_beta(idata):
    """
    Posterior mean of `beta`, as an (N_nights, L) array.

    Accumulated one chain at a time in float64. The chains are float32 and
    there are ~144,000 draws of a beta_0 as large as the data itself (~780),
    so a float32 reduction over the whole sample axis loses digits exactly
    where this figure reads them -- the night-to-night spread is orders of
    magnitude smaller than the coefficients themselves. Per-chain slicing also
    keeps peak memory at one chain (~50 MB) rather than the whole array
    widened to float64.
    """
    beta     = idata.posterior['beta']
    n_chains = beta.sizes['chain']
    n_draws  = beta.sizes['draw']

    total = np.zeros(beta.shape[2:], dtype=np.float64)
    for c in range(n_chains):
        total += np.asarray(beta.isel(chain=c).values, dtype=np.float64).sum(axis=0)

    return total / (n_chains * n_draws)


def plot_corner_legendre(idata, night_ids=None, n_coeff=None, savefig=False,
                         savepath=None, show=False):
    """
    Corner plot of the fitted Legendre background coefficients, night by night
    (`__make_paper_figs/legendre_coeff_corner.py`).

    Each night's `beta` posterior is collapsed to its MEAN -- one number per
    (night, coefficient) -- and the resulting (N_nights, L) table is corner
    plotted, so the points in every panel are NIGHTS, not posterior draws.
    This deliberately throws the per-night posterior uncertainty away and shows
    the opposite thing: how much the background SHAPE itself moves across the
    season, and which coefficients move together when it does. A tight blob
    means the season's backgrounds are all essentially the same curve; an
    elongated one means two coefficients trade off against each other night to
    night.

    Points are coloured by the night's date, so a seasonal drift in any
    coefficient (or a handful of outlying nights) is visible without labelling
    40-odd points.

    Parameters
    ----------
    idata : arviz.InferenceData
        Fitted posterior, with a per-night `beta`.
    night_ids : sequence of str, optional
        The night ids behind `beta`'s night axis, in fit order (i.e.
        `data_dict['night_ids']`). Used only for the date colour code; without
        them the points are coloured by night index instead. Length is checked
        against the posterior, since a mismatched list would silently mis-date
        every point.
    n_coeff : int, optional
        How many of the L coefficients to plot, lowest-order first. None (the
        default) plots all of them, which at L = 24 is a 24 x 24 wall of
        panels -- pass 6-8 for a legible sub-figure of the modes that carry
        most of the background's shape.
    """
    # The grid is n_coeff panels across, so the figure has to grow with it or
    # the panels come out postage-stamp sized. At L = 24 this is a ~26 inch
    # figure, which is already ~4000 px at the dpi below -- hence 150 rather
    # than the 300 the smaller figures use.
    PANEL_INCHES = 1.1
    DPI          = 150
    # Histogram bins on the diagonal. There are only ~40 nights, so corner's
    # default (20) leaves most bins holding one night and the diagonal reads
    # as noise.
    N_BINS     = 12
    POINT_SIZE = 16
    CMAP       = 'viridis'

    beta_mean = _posterior_mean_beta(idata)
    n_nights  = beta_mean.shape[0]

    x    = beta_mean[:, :n_coeff] if n_coeff else beta_mean
    k    = x.shape[1]

    if night_ids is not None and len(night_ids) != n_nights:
        raise ValueError(
            f"{len(night_ids)} night ids but the posterior has {n_nights} "
            "nights -- the night list doesn't belong to this posterior."
        )

    if night_ids is None:
        t        = np.arange(n_nights, dtype=float)
        cb_label = f"Night index — {n_nights} nights"
        date_cb  = False
    else:
        t        = mdates.date2num(night_dates(night_ids))
        cb_label = f"Night (UTC) — {n_nights} nights"
        date_cb  = True

    # corner needs at least as many points as dimensions. A fit with fewer
    # nights than coefficients (a half-split, say) can't be corner-plotted at
    # all -- skip the figure rather than take the postprocessing task down
    # with it.
    if n_nights <= k:
        print(f"Skipping the Legendre corner plot: {n_nights} nights is too few "
              f"for {k} coefficients. Pass n_coeff < {n_nights} to plot a subset.")
        return

    norm = Normalize(vmin=t.min(), vmax=t.max())

    size = PANEL_INCHES * k
    fig  = plt.figure(figsize=(size, size))

    # Nights are plotted as plain points: with ~40 of them in L dimensions
    # there is nothing for corner's contours or density image to estimate, and
    # asking for them produces either an error or a misleading smooth blob.
    # corner draws the panel frames, limits and diagonal histograms; the
    # off-diagonal points are scattered on below, because corner draws its
    # points with `Axes.plot`, which takes one colour for the whole series and
    # so can't carry the date colour code.
    corner.corner(
        x,
        labels=[rf"$\beta_{{{i}}}$" for i in range(k)],
        fig=fig,
        bins=N_BINS,
        plot_datapoints=False,
        plot_density=False,
        plot_contours=False,
        hist_kwargs=dict(color="0.35", lw=1.0),
        label_kwargs=dict(fontsize=11),
        labelpad=0.08,
    )

    # corner builds the grid with plt.subplots(k, k), so its axes come back in
    # row-major order: axes[i, j] has coefficient j on x and coefficient i on
    # y, and only the lower triangle (j < i) is drawn.
    axes = np.array(fig.axes).reshape((k, k))
    for i in range(k):
        for j in range(i):
            axes[i, j].scatter(x[:, j], x[:, i], c=t, cmap=CMAP, norm=norm,
                               s=POINT_SIZE, linewidths=0.3,
                               edgecolors='white', zorder=3)

    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=8)

    # Up in the empty top-right triangle, where it costs no panel space. With
    # dates, the bar doubles as the season's calendar span.
    cax = fig.add_axes([0.62, 0.90, 0.30, 0.012])
    cb  = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP),
                       cax=cax, orientation='horizontal')
    cb.set_label(cb_label, fontsize=12)
    if date_cb:
        cb.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    cb.ax.tick_params(labelsize=10)

    fig.suptitle("Per-night posterior-mean Legendre coefficients",
                 fontsize=16, x=0.62, y=0.96, ha='center')

    if savefig:
        fig.savefig(os.path.join(savepath, "legendre_corner.png"),
                    dpi=DPI, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_night(idata, z, data_dict, ndx, mu_bg, pointing, sigma_clean,
                savefig=False, savepath=None, show=False):
    """
    Three-panel figure for one night (`__make_paper_figs/get_night_examples.py`):
    data with the fitted background, background-subtracted residuals, and the
    posterior soft labels, sharing a time axis in seconds.

    The panels carry the posterior's own spread rather than a sample of draws:
    the background is a `N_SIGMA_BG`-sigma band, the residuals and soft labels
    carry `N_SIGMA_ERRORBARS`-sigma error bars. Every panel draws its base
    series (black data, grey residuals, black soft labels) and then overlays
    the model-only / SSINS-only / both RFI colour code on top, so a flagged
    sample can be followed down the panels by eye.

    Parameters
    ----------
    idata : arviz.InferenceData
        Fitted posterior; used only for the per-night r-hat in the title.
    z : array-like, shape (S, N, T)
        Per-draw residuals, from `postprocessing_utils.get_bg_residuals`.
    data_dict : dict
        Padded per-night data, with the label arrays stage 3 adds to it
        (`hard_labels`, `soft_labels`, `soft_labels_std`).
    ndx : int
        Night index into the fitted night list.
    mu_bg : array-like, shape (S, N, T)
        Per-draw Legendre background, from `get_bg_residuals`.
    pointing : int
        Pointing, for the title.
    sigma_clean : array-like, shape (N,)
        Per-night analytic thermal noise, for the title.
    """

    rhat_beta = az.rhat(idata, var_names=["beta"])["beta"].values
    rhat_per_night = rhat_beta.mean(axis=-1)

    stop  = int(data_dict['lengths'][ndx])
    times = np.asarray(data_dict['times'])[ndx, :stop]
    x     = (times - times[0]) * 24 * 3600

    y          = np.asarray(data_dict['y_pad'])[ndx, :stop]
    ssins_mask = np.asarray(data_dict['f_pad'])[ndx, :stop].astype(bool)
    model_mask = np.asarray(data_dict['hard_labels'])[ndx, :stop].astype(bool)

    bg_mean, bg_std = _moments(mu_bg, ndx, stop)
    z_mean,  z_std  = _moments(z,     ndx, stop)

    label_mean = np.asarray(data_dict['soft_labels'])[ndx, :stop]
    label_std  = np.asarray(data_dict['soft_labels_std'])[ndx, :stop]

    masks = rfi_masks(model_mask, ssins_mask)

    # The lower two panels' base series are dense enough that a black/grey dot
    # shows through the thin points of a red star drawn on top of it, making
    # the star read as a dot. Drop the base marker wherever a star goes; the
    # other two categories' markers are solid and cover their own dots.
    unstarred = ~masks['both']

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.5, 0.8]}
    )

    # --- data + n-sigma background band ---------------------------------
    # Drawn before the band so that "Clean" leads the legend.
    axes[0].scatter(x, y, label="Clean", s=5, color="black")

    axes[0].fill_between(x, bg_mean - N_SIGMA_BG * bg_std,
                            bg_mean + N_SIGMA_BG * bg_std,
                         color='grey', alpha=0.3,
                         label=fr"Background ({N_SIGMA_BG}$\sigma$)")
    axes[0].plot(x, bg_mean, color='grey', linewidth=1.5)

    scatter_rfi(axes[0], x, y, masks)
    axes[0].legend()
    axes[0].set_ylabel("SSINS (arb. units)")
    axes[0].set_title(
        f"Pointing {pointing}, night {ndx+1} | "
        f"mean r-hat={rhat_per_night[ndx]:.3f} | "
        f"noise std={sigma_clean[ndx]:.3f}"
    )

    # --- residuals, n-sigma error bars ----------------------------------
    axes[1].errorbar(
        x[unstarred], z_mean[unstarred],
        yerr=N_SIGMA_ERRORBARS * z_std[unstarred],
        fmt="o", markersize=3, color="grey",
        ecolor="grey", elinewidth=0.8,
        label=fr"Error bars: {N_SIGMA_ERRORBARS}$\sigma$" "\n" "(this and lower panel)",
    )
    errorbar_rfi(axes[1], x, z_mean, N_SIGMA_ERRORBARS * z_std, masks,
                 star_size=ERRORBAR_STAR_SIZE)
    axes[1].axhline(0, linestyle="--", linewidth=1, color='grey')
    axes[1].set_ylabel("Residuals")

    leg = axes[1].legend(loc="upper left", fontsize=10, frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.7)
    leg.get_frame().set_edgecolor("none")

    # --- soft labels, n-sigma error bars --------------------------------
    axes[2].errorbar(
        x[unstarred], label_mean[unstarred],
        yerr=N_SIGMA_ERRORBARS * label_std[unstarred],
        fmt="o", markersize=3, color="black",
        ecolor="black", elinewidth=0.8,
    )
    errorbar_rfi(axes[2], x, label_mean, N_SIGMA_ERRORBARS * label_std, masks,
                 star_size=ERRORBAR_STAR_SIZE)
    axes[2].axhline(0.5, linestyle="--", linewidth=1, color='black')
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["0 - clean", "1 - RFI"])
    axes[2].set_ylabel("Soft label")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    if savefig:
        visdir = os.path.join(savepath, "visualize/")
        os.makedirs(visdir, exist_ok=True)
        plt.savefig(os.path.join(visdir, f"visualize_night{ndx:02d}.png"),
                    dpi=300,bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_chisq(chi2_median, chi2_lo, chi2_hi, savepath=None, savefig=False, show=False):
    """
    """
    nights = np.arange(chi2_median.size)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.errorbar(
        nights, chi2_median,
        yerr=[chi2_median - chi2_lo, chi2_hi - chi2_median],
        fmt='o', capsize=3, linewidth=1, label='median + 94% HDI'
    )
    ax.axhline(1.0, linestyle='--', color='red', linewidth=1, label=r'ideal (reduced $\chi^2$=1)')
    ax.set_xlabel("Night index")
    ax.set_ylabel(r"Reduced $\chi^2$")
    ax.set_title(r"Reduced $\chi^2$ per night (clean samples only)")
    ax.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig(os.path.join(savepath, "clean_chi2_per_night.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_nll(nll_median, nll_lo, nll_hi, savepath=None, savefig=False, show=False):
    """
    """
    nights = np.arange(nll_median.size)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.errorbar(
        nights, nll_median,
        yerr=[nll_median - nll_lo, nll_hi - nll_median],
        fmt='o', capsize=3, linewidth=1, label='median + 94% HDI'
    )
    ax.set_xlabel("Night index")
    ax.set_ylabel("Mean negative log-pdf")
    ax.set_title("SkewCauchy fit to RFI residuals per night")
    ax.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig(os.path.join(savepath, "rfi_nll_per_night.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_clean_rfi_histograms(z, data_dict, samples, sigma_clean, savepath=None, savefig=False, show=False):
    """
    Histograms of the residuals the model called clean and RFI, each with the
    component it was assigned to drawn over it
    (`__make_paper_figs/show_residuals.ipynb`).

    Two panels side by side, both on a log density axis. The clean panel
    carries one Normal(0, sigma_clean) curve per night -- the analytic thermal
    noise, which is not fitted, so the agreement is a check rather than a fit.
    The RFI panel carries the posterior of the SkewCauchy component as a
    pointwise band: the spread of `N_SIGMA_PDF` standard deviations taken in
    LOG density, since that is the axis the curve is read on, with the
    geometric mean drawn through it.

    Parameters
    ----------
    z : array-like, shape (S, N, T)
        Per-draw residuals.
    data_dict : dict
        Output of prepare_multiple_nights, plus stage 3's 'hard_labels'.
    samples : dict
        Posterior samples from `extract_samples` (needs loc_rfi, scale_rfi).
    sigma_clean : array-like, shape (N,)
        Per-night thermal noise standard deviation.
    savepath : str, optional
        Directory to save the figure in.
    savefig : bool, optional
        Whether to save the figure to disk.
    show : bool, optional
        Whether to call plt.show().
    """
    A_RFI       = 0.99
    N_SIGMA_PDF = 10
    # Draws to build the RFI band from. The band is a pointwise mean and std
    # over draws, so a few thousand of the ~144,000 draws fix it to well
    # inside the linewidth -- while evaluating every draw on the x grid would
    # allocate an (S, 1000) float array of order 1 GB inside a stage-3 task.
    MAX_PDF_DRAWS = 2000

    # Flatten residuals and labels across nights
    z_mean = np.mean(z, axis=0)  # (N, T)

    clean_samples = []
    rfi_samples   = []

    for plot_night in range(len(data_dict['lengths'])):
        stop    = int(data_dict['lengths'][plot_night])
        avg_z   = np.asarray(z_mean[plot_night, :stop])
        hlabels = np.asarray(data_dict['hard_labels'][plot_night, :stop]).astype(bool)

        finite = np.isfinite(avg_z)
        clean_samples.extend(avg_z[finite & ~hlabels])
        rfi_samples.extend(avg_z[finite & hlabels])

    clean_samples = np.asarray(clean_samples)
    rfi_samples   = np.asarray(rfi_samples)

    loc_full   = np.asarray(samples['loc_rfi']).ravel()
    scale_full = np.asarray(samples['scale_rfi']).ravel()
    if loc_full.size > MAX_PDF_DRAWS:
        pick       = np.random.default_rng(0).choice(loc_full.size, MAX_PDF_DRAWS, replace=False)
        loc_full   = loc_full[pick]
        scale_full = scale_full[pick]

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.5))

    # ---- clean residuals vs. the analytic thermal noise ------------------
    axes[0].hist(clean_samples, density=True, bins=40,
                 color='black',
                 histtype="step",
                 linewidth=2,
                )
    axes[0].set_xlabel("Residuals - samples marked as clean")

    x_pdf = np.linspace(clean_samples.min(), clean_samples.max(), 500)
    for s in np.asarray(sigma_clean):
        axes[0].plot(x_pdf, norm.pdf(x_pdf, loc=0, scale=s),
                     color=sns.color_palette()[2], alpha=0.1)
    axes[0].set_yscale("log")
    axes[0].set_ylim(5e-4, 1.5)
    axes[0].set_ylabel("Probability density")

    # ---- RFI residuals vs. the fitted SkewCauchy -------------------------
    axes[1].hist(rfi_samples, density=True, bins=55,
                 color='black',
                 histtype="step",
                 linewidth=2,
                )
    axes[1].set_xlabel("Residuals - samples marked as RFI")
    axes[1].set_yscale("log")

    x_pdf = np.linspace(rfi_samples.min(), rfi_samples.max(), 1000)

    # The pdf for every kept draw -> (n_draws, n_x)
    curves = skewcauchy.pdf(
        x_pdf[None, :],
        a=A_RFI,
        loc=loc_full[:, None],
        scale=scale_full[:, None],
    )

    log_c   = np.log(np.clip(curves, 1e-300, None))
    mu, sd  = log_c.mean(axis=0), log_c.std(axis=0)
    lo, hi  = np.exp(mu - N_SIGMA_PDF * sd), np.exp(mu + N_SIGMA_PDF * sd)
    med     = np.exp(mu)

    axes[1].fill_between(x_pdf, lo, hi, color=sns.color_palette()[2], alpha=0.3, linewidth=0)
    axes[1].plot(x_pdf, med, color=sns.color_palette()[2], linewidth=1.2)
    axes[1].set_ylim(1e-6, 0.4)

    plt.tight_layout()
    if savefig:
        plt.savefig(os.path.join(savepath, "residuals.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def compare_model_pdfs(samples, sigma_clean, savefig=False, savepath=None, show=False):
    """
    """
    x = np.linspace(-10, 10, 1000)

    plt.figure(figsize=(8,4))

    for i in range(1000):
        label = 'SkewCauchy random draw' if i==0 else None
        idx = np.random.choice(samples['loc_rfi'].size)
        loc_rfi   = samples['loc_rfi'][idx]
        scale_rfi = samples['scale_rfi'][idx]
        skew_cauchy = mixture_model.SkewCauchy(loc_rfi, scale_rfi, a_rfi=0.99)
        pdf = np.exp(np.array(skew_cauchy.log_prob(jnp.array(x))))
        plt.plot(x, pdf, linewidth=2, label=label,
                alpha=0.05, color=sns.color_palette()[2])

    for sc in sigma_clean:
            plt.plot(x, norm.pdf(x, loc=0, scale=sc), color=sns.color_palette()[0], linewidth=1, alpha=0.2)
        
    plt.legend()
    # plt.yscale('log')
    # plt.ylim(1e-2,2)
    if savefig:
        plt.savefig(os.path.join(savepath, "compare_clean_rfi_histograms.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
