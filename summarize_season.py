import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

import cli
import hdf5_utils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8-white')  # the style the paper figures are drawn in

ABS_DIR = "/oscar/data/jpober/jmduchar/Research/BayesRFI/"

TV_SUBBANDS   = ['subTV', 'TV6', 'TV7', 'TV8', 'TV9']
POINTINGS     = [0, 1, 2, 3, 4]
POLARIZATIONS = ['xx', 'yy']

# MWA GRIDNUM -> pointings east(+)/west(-) convention
POINTING_DICT  = {0: 0, 1: -1, 2: 1, 3: -2, 4: 2}
POINTING_ORDER = [3, 1, 0, 2, 4]
MARKERS        = {'xx': 'o', 'yy': 'X'}

# This script isn't parallel over anything --> it reads the whole season at once
N_JOBS = 1


def summarize_season(run_dir, half=None):
    """
    Produce the season-level summary plots for one run.

    Reads the run's shared results file into a single tidy dataframe and writes
    heatmaps of the global posterior parameters, per-night reduced chi-squared,
    posterior-vs-empirical RFI fraction, and the RFI-responsibility time series.

    Parameters
    ----------
    run_dir : str
        Run directory holding `processed/season_results.h5`.
    half : int, optional
        0 or 1. If given, reads that half-split run's
        `processed/half{N}_season_results.h5` instead of the full-season file,
        and writes its plots to a correspondingly prefixed subdirectory.
    """
    prefix   = f"half{half}_" if half is not None else ""
    base_dir = os.path.join(run_dir, "processed")
    save_dir = os.path.join(base_dir, f"{prefix}seasonal_statistics")
    h5_path  = os.path.join(base_dir, prefix + hdf5_utils.HDF5_FILENAME)

    os.makedirs(save_dir, exist_ok=True)
    print(f"Reading {h5_path}")
    print(f"Saving plots to {save_dir}")

    # ----------------------------
    # Load everything from the shared HDF5 file
    # ----------------------------

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 results file not found: {h5_path}")

    df = hdf5_utils.read_season_results(h5_path)

    # One row per (pol, tv_subband, pointing)
    combo_df = df.drop_duplicates(subset=['pol', 'tv_subband', 'pointing']).copy()

    # ----------------------------
    # w_rfi heatmap vs pointing/subband
    # ----------------------------

    for pol in POLARIZATIONS:
        fig, ax = plt.subplots(figsize=(8, 5))
        subset = combo_df[combo_df['pol'] == pol].copy()
        subset['w_rfi_median'] = 1 - subset['w_clean_median']
        pivot  = subset.pivot(index='pointing', columns='tv_subband', values='w_rfi_median')
        pivot  = pivot.reindex(columns=TV_SUBBANDS, index=POINTINGS)
        sns.heatmap(pivot, annot=True, fmt='.3f', ax=ax, cmap='Reds', vmin=0, vmax=0.133)
        ax.set_title(f"w_rfi median — pol={pol}")
        ax.set_xlabel("TV subband")
        ax.set_ylabel("Pointing")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"w_rfi_vs_pointing_subband_{pol}.png"), dpi=300, bbox_inches='tight')
        plt.show()

    # ----------------------------
    # loc_rfi heatmap vs pointing/subband
    # ----------------------------

    for pol in POLARIZATIONS:
        fig, ax = plt.subplots(figsize=(8, 5))
        subset = combo_df[combo_df['pol'] == pol]
        pivot  = subset.pivot(index='pointing', columns='tv_subband', values='loc_rfi_median')
        pivot  = pivot.reindex(columns=TV_SUBBANDS, index=POINTINGS)
        sns.heatmap(pivot, annot=True, fmt='.3f', ax=ax, cmap='Reds',
                    vmin=0, vmax=0.154,
                   )
        ax.set_title(f"loc_rfi median — pol={pol}")
        ax.set_xlabel("TV subband")
        ax.set_ylabel("Pointing")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"loc_rfi_vs_pointing_subband_{pol}.png"), dpi=300, bbox_inches='tight')
        plt.show()

    # ----------------------------
    # scale_rfi heatmap vs pointing/subband
    # ----------------------------

    for pol in POLARIZATIONS:
        fig, ax = plt.subplots(figsize=(8, 5))
        subset = combo_df[combo_df['pol'] == pol]
        pivot  = subset.pivot(index='pointing', columns='tv_subband', values='scale_rfi_median')
        pivot  = pivot.reindex(columns=TV_SUBBANDS, index=POINTINGS)
        sns.heatmap(pivot, annot=True, fmt='.3f', ax=ax, cmap='Reds',
                   vmin=0, vmax=2.68,
                   )
        ax.set_title(f"scale_rfi median — pol={pol}")
        ax.set_xlabel("TV subband")
        ax.set_ylabel("Pointing")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"scale_rfi_vs_pointing_subband_{pol}.png"), dpi=300, bbox_inches='tight')
        plt.show()

    # ----------------------------
    # w_rfi vs subband, all pointings overlaid
    # ----------------------------
    
    series  = [(p, pol) for p in POINTING_ORDER for pol in MARKERS]
    colors  = np.array(sns.color_palette("Paired")).reshape(-1, 2, 3)[:, ::-1].reshape(-1, 3)
    offsets = dict(zip(POINTING_ORDER, np.linspace(-0.36, 0.36, len(POINTING_ORDER))))

    fig, ax = plt.subplots(figsize=(9, 3.5))

    for cdx, (p, pol) in enumerate(series):
        subset = combo_df[(combo_df['pol'] == pol) &
                          (combo_df['pointing'] == p)].sort_values('tv_order')
        med = 1 - subset['w_clean_median'].to_numpy()
        lo  = 1 - subset['w_clean_hdi_94_hi'].to_numpy()   # flipping 1-x swaps lo/hi
        hi  = 1 - subset['w_clean_hdi_94_lo'].to_numpy()
        ax.errorbar(
            np.arange(len(subset)) + offsets[p], med,
            yerr=[med - lo, hi - med],
            marker=MARKERS[pol], color=colors[cdx],
            markersize=6,
            markeredgecolor='white', markeredgewidth=0.6,  # separates marker from bar
            capsize=2, elinewidth=0.9,
            linestyle='none',
            zorder=3,                                      # markers above the bars
            label=f"{POINTING_DICT[p]} {pol}",
        )

    for i in range(len(TV_SUBBANDS) - 1):                  # group separators
        ax.axvline(i + 0.5, color='0.9', lw=0.8, zorder=0)

    ax.set_xticks(np.arange(len(TV_SUBBANDS)))
    ax.set_xticklabels(TV_SUBBANDS)
    ax.set_xlabel("TV sub-band")
    ax.set_ylabel("RFI seasonal occupancy")
    ax.legend(ncol=5, fontsize=9, frameon=False,
              loc='upper center', bbox_to_anchor=(0.5, 1.3))
    ax.grid(color='0.9', linewidth=0.6, axis='y', zorder=0)
    ax.set_axisbelow(True)   # keeps bars drawn on top of the grid

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "w_rfi_vs_subband_by_pointing.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # ----------------------------
    # Reduced chi-squared per night, grid over subband x pointing
    # ----------------------------

    night_df = df.drop_duplicates(subset=['pol', 'tv_subband', 'pointing', 'night'])

    fig, axes = plt.subplots(len(TV_SUBBANDS), len(POINTINGS),
                              figsize=(4 * len(POINTINGS), 3 * len(TV_SUBBANDS)),
                              sharex=False, sharey=True)

    for i, tv in enumerate(TV_SUBBANDS):
        for j, p in enumerate(POINTINGS):
            ax = axes[i][j]
            for pol, color in zip(POLARIZATIONS, sns.color_palette()[:2]):
                subset = night_df[
                    (night_df['tv_subband'] == tv) &
                    (night_df['pointing']   == p)  &
                    (night_df['pol']        == pol)
                ].sort_values('night')
                if subset.empty:
                    continue
                ax.errorbar(
                    subset['night'],
                    subset['chisq_median'],
                    yerr=[subset['chisq_median'] - subset['chisq_hdi_94_lo'],
                          subset['chisq_hdi_94_hi'] - subset['chisq_median']],
                    fmt='.', capsize=2, linewidth=0.8,
                    label=pol, color=color, alpha=0.8
                )
            ax.axhline(1.0, linestyle='--', color='red', linewidth=0.8)
            ax.set_title(f"{tv} p{p}", fontsize=8)
            if j == 0:
                ax.set_ylabel(r"Reduced $\chi^2$")
            if i == len(TV_SUBBANDS) - 1:
                ax.set_xlabel("Night")

    handles = [plt.Line2D([0], [0], color=c, marker='o', label=pol)
               for pol, c in zip(POLARIZATIONS, sns.color_palette()[:2])]
    fig.legend(handles=handles, loc='upper right')
    plt.suptitle(r"Reduced $\chi^2$ per night across all configurations", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "chisq_per_night.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # ----------------------------
    # Posterior w_rfi vs empirical RFI fraction (hard and soft labels)
    # ----------------------------

    label_frac = (
        df.groupby(['pol', 'tv_subband', 'pointing'])[['hard_label', 'soft_label']]
        .mean()
        .rename(columns={'hard_label': 'hard_label_frac', 'soft_label': 'soft_label_frac'})
        .reset_index()
    )

    df_posterior_wrfi = combo_df[
        ['pol', 'tv_subband', 'pointing', 'w_clean_median', 'w_clean_hdi_94_lo', 'w_clean_hdi_94_hi']
    ].copy()
    df_posterior_wrfi['w_rfi_posterior'] = 1 - df_posterior_wrfi['w_clean_median']
    df_posterior_wrfi['w_rfi_lo']        = 1 - df_posterior_wrfi['w_clean_hdi_94_hi']  # bounds flip
    df_posterior_wrfi['w_rfi_hi']        = 1 - df_posterior_wrfi['w_clean_hdi_94_lo']
    df_merged = label_frac.merge(df_posterior_wrfi, on=['pol', 'tv_subband', 'pointing'])

    for ldx, label in enumerate(['hard_label_frac', 'soft_label_frac']):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)

        for ax, pol in zip(axes, POLARIZATIONS):
            subset = df_merged[df_merged['pol'] == pol]

            for tv, marker in zip(TV_SUBBANDS, ['o', 's', '^', 'D', 'v']):
                s = subset[subset['tv_subband'] == tv]
                ax.errorbar(
                    s[label],
                    s['w_rfi_posterior'],
                    yerr=[s['w_rfi_posterior'] - s['w_rfi_lo'],
                        s['w_rfi_hi'] - s['w_rfi_posterior']],
                    fmt=marker, capsize=3, label=tv, markersize=7
                )

            # y=x line
            lims = [0, max(df_merged[label].max(), df_merged['w_rfi_posterior'].max()) * 1.05]
            ax.plot(lims, lims, 'k--', linewidth=1, label='perfect agreement')
            ax.set_xlabel(f"Empirical RFI fraction (from {label.replace('_frac', 's')})")
            ax.set_ylabel("Posterior w_rfi (from MCMC)")
            ax.set_title(f"pol={pol}")
            ax.legend(fontsize=8)

        plt.suptitle("Posterior w_rfi vs empirical RFI fraction (HARD LABELS)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"posterior_vs_empirical_rfi_{'HARD' if ldx==0 else 'SOFT'}.png"), dpi=300, bbox_inches='tight')
        plt.show()

    # ----------------------------
    # RFI responsibility over time, TV7
    # ----------------------------

    for pol in POLARIZATIONS:
        subset = df[
            (df['pol'] == pol) & (df['tv_subband'] == 'TV7') & np.isfinite(df['soft_label'])
        ].sort_values('time_jd')
        all_times  = subset['time_jd'].to_numpy()
        all_labels = subset['soft_label'].to_numpy()

        fig, ax = plt.subplots(figsize=(18, 3))
        ax.scatter(all_times, all_labels, s=0.5, alpha=0.3, rasterized=True)
        ax.set_xlabel("Julian Date")
        ax.set_ylabel("Soft label (p_rfi)")
        ax.set_title(f"RFI responsibility over time — TV7, pol={pol}")
        plt.tight_layout()
        plt.show()


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    parser = cli.build_parser(
        description="Pipeline stage 4: season-level summary plots from one run's "
                    "aggregated results file. Single-task stage: job_index is always 0.",
        n_jobs=N_JOBS,
        epilog="Examples:\n"
               "  python summarize_season.py 0\n"
               "  python summarize_season.py 0 --run-dir out_chrono_split --half 0",
    )
    cli.add_run_dir(parser, default="out",
                    help="run directory to read processed/ from and write plots into")
    cli.add_half(parser)
    args = cli.parse_args(parser)

    run_dir = cli.resolve_run_dir(args.run_dir, ABS_DIR)

    cli.print_config(
        "Seasonal statistics",
        job_index = args.job_index,
        half      = args.half,
        run_dir   = run_dir,
    )

    summarize_season(run_dir, half=args.half)
