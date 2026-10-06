import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

import glob

import arviz as az
import cli
import hdf5_utils
import pandas as pd
import plotting_utils as plu
import postprocessing_utils as pu

# ----------------------------
# Globals
# ----------------------------

ABS_DIR       = os.environ.get("BAYESRFI_ABS_DIR", "/oscar/data/jpober/jmduchar/Research/BayesRFI/")
TV_SUBBANDS   = ['subTV', 'TV6', 'TV7', 'TV8', 'TV9']
POINTINGS     = [0, 1, 2, 3, 4]
POLARIZATIONS = ['xx', 'yy']
L             = 24

# Keep every SOFT_LABEL_THIN-th draw of each chain
SOFT_LABEL_THIN = 1

JOB_MAP = {
    idx: (L, tv, p, pol)
    for idx, (tv, p, pol) in enumerate(
        (tv, p, pol)
        for tv   in TV_SUBBANDS
        for p    in POINTINGS
        for pol  in POLARIZATIONS
    )
}

# ----------------------------
# Functions
# ----------------------------


def identify_stuck_chains(idata, savepath=None):
    """
    Identify and remove stuck MCMC chains based on per-chain std of key global parameters.
    Computes convergence diagnostics before and after chain removal and optionally saves
    a summary to YAML.

    Parameters
    ----------
    idata : arviz.InferenceData
        MCMC output.
    savepath : str, optional
        If provided, saves diagnostics YAML to this directory.

    Returns
    -------
    idata : arviz.InferenceData
        InferenceData with stuck chains removed.
    """

    idata_summary = az.summary(idata)
    idata_summary['r_hat'] = pd.to_numeric(idata_summary['r_hat'], errors='coerce')

    n_chains_before = int(idata.posterior.dims['chain'])
    n_draws         = int(idata.posterior.dims['draw'])

    # ---- Identify stuck chains ----------------------------------------
    key_params = ['w_rfi', 'loc_rfi', 'scale_rfi']
    chain_stds = {p: idata.posterior[p].std(dim='draw').values for p in key_params}

    bad_chains = set()
    stuck_details = []
    for p, stds in chain_stds.items():
        global_std = float(idata.posterior[p].std(dim=('chain', 'draw')))
        for i, s in enumerate(stds):
            if s < 1e-3 * global_std:
                bad_chains.add(i)
                detail = (f"Chain {i} stuck on '{p}': "
                          f"chain std={s:.2e}, global std={global_std:.2e}")
                stuck_details.append(detail)
                print(f"WARNING: {detail}")

    bad_chains  = sorted(bad_chains)
    good_chains = [i for i in range(n_chains_before) if i not in bad_chains]

    if bad_chains:
        print(f"\nRemoving {len(bad_chains)} stuck chain(s): {bad_chains}")
        print(f"Keeping {len(good_chains)} chain(s).\n")
        idata = idata.sel(chain=good_chains)
    else:
        print("No stuck chains detected.\n")

    # ---- Convergence diagnostics after trimming -----------------------
    idata_summary_trimmed = az.summary(idata)
    idata_summary_trimmed['r_hat'] = pd.to_numeric(idata_summary_trimmed['r_hat'], errors='coerce')

    n_high_rhat    = int((idata_summary_trimmed['r_hat'] > 1.01).sum())
    max_rhat       = float(idata_summary_trimmed['r_hat'].max())
    median_rhat    = float(idata_summary_trimmed['r_hat'].median())
    mean_ess_bulk  = float(pd.to_numeric(idata_summary_trimmed['ess_bulk'], errors='coerce').mean())
    mean_ess_tail  = float(pd.to_numeric(idata_summary_trimmed['ess_tail'],  errors='coerce').mean())

    # Diagnostics
    n_divergences  = int(idata.sample_stats.diverging.values.sum()) if hasattr(idata, 'sample_stats') else None

    diagnostics = {
        'n_chains_before'      : n_chains_before,
        'n_chains_after'       : len(good_chains),
        'n_stuck_chains'       : len(bad_chains),
        'stuck_chains'         : bad_chains,
        'stuck_details'        : stuck_details,
        'n_draws_per_chain'    : n_draws,
        'n_high_rhat_after'    : n_high_rhat,
        'max_rhat_after'       : round(max_rhat, 4),
        'median_rhat_after'    : round(median_rhat, 4),
        'mean_ess_bulk_after'  : round(mean_ess_bulk, 1),
        'mean_ess_tail_after'  : round(mean_ess_tail, 1),
        'n_divergences'        : n_divergences,
    }

    print("---- Convergence diagnostics (after trimming) ----")
    for k, v in diagnostics.items():
        print(f"  {k:30s}: {v}")
    print()

    return idata, diagnostics


def postprocess_chains(L, tv_subband, pointing, polarization, run_dir, abs_dir=ABS_DIR, half=None):
    """
    Full postprocessing pipeline for a single MCMC run:
        1. Load most recent matching chains file
        2. Identify and remove stuck chains
        3. Plot convergence diagnostics
        4. Compute and save background, residuals, and classification labels
        5. Plot per-night predictions
        6. Compute and plot chi-squared and RFI neg-log-pdf
        7. Save posterior summary

    Parameters
    ----------
    L            : int    Number of Legendre coefficients.
    tv_subband   : str    TV sub-band identifier (e.g. 'TV7').
    pointing     : int    Pointing index.
    polarization : str    Polarization string (e.g. 'xx').
    abs_dir      : str    Absolute path to project root directory.
    run_dir      : str, optional  Directory tree to read chains from / write processed
                   output to. Defaults to `out/` (or `out_halfsplit/` if `half` is given).
    half         : int, optional  0 or 1. If given, reads the corresponding half-split
                   run from `out_halfsplit/` instead of the full run from `out/`.
    """

    # ---- Find chains file ---------------------------------------------
    pattern = os.path.join(run_dir, f"chains/chains_{tv_subband}_p{pointing}_pol{polarization}*.nc")
    matching_files = glob.glob(pattern)

    # Half-split runs are tagged "_half{0,1}_" in the filename
    if half is not None:
        matching_files = [f for f in matching_files if f"_half{half}_" in os.path.basename(f)]
    else:
        matching_files = [f for f in matching_files if "_half" not in os.path.basename(f)]

    if not matching_files:
        raise FileNotFoundError(f"No chains file found matching pattern: {pattern}"
                                 + (f" (half={half})" if half is not None else ""))

    # If multiple files, use the most recent one
    output_file       = max(matching_files, key=os.path.getmtime)
    output_timestring = output_file.split("chains_")[-1].split(".")[0]
    print(f"Using latest chains file: {output_file}")

    notes_path = os.path.join(run_dir, f"chains/chains_{output_timestring}.txt")
    if os.path.exists(notes_path):
        with open(notes_path, "r") as f:
            print(f.read())

    # ---- Load the exact night_ids the chain was fit on -----------------
    nights_path = os.path.join(run_dir, f"chains/chains_{output_timestring}_nights.txt")
    if os.path.exists(nights_path):
        with open(nights_path, "r") as f:
            keep_night_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(keep_night_ids)} night(s) from {nights_path}")
    elif half is not None:
        raise FileNotFoundError(
            f"No saved night list found at {nights_path}, but half={half} was requested."
        )
    else:
        keep_night_ids = None
        print(f"No saved night list found at {nights_path}; using the full (rejection-filtered) "
              "season. Only correct if this chain was fit on the full season.")

    savepath = os.path.join(run_dir, f"processed/processed_{output_timestring}/")
    os.makedirs(savepath, exist_ok=True)
    print("Saving processed plots and metrics to:", savepath)

    # ---- Load and clean chains ----------------------------------------
    idata, diagnostics = identify_stuck_chains(
        az.from_netcdf(output_file),
        savepath=savepath,
    )

    # ---- Load input data -----------------------------------------------
    sigma_clean, data_dict, rejected_obsids = pu.load_input_data(
        L, abs_dir, tv_subband, pointing, polarization, keep_night_ids=keep_night_ids
    )

    # ---- Convergence plots --------------------------------------------
    print("\nPlotting convergence metrics...")
    idata_summary = az.summary(idata)
    plu.plot_convergence_metrics(idata_summary, savefig=True, savepath=savepath)
    plu.plot_trace_corner(idata,                savefig=True, savepath=savepath)
    plu.plot_corner_mu_sigma(idata,             savefig=True, savepath=savepath)
    plu.plot_corner_legendre(idata, night_ids=data_dict['night_ids'],
                             savefig=True, savepath=savepath)

    # ---- Predictions --------------------------------------------------
    print("\nComputing background, residuals, and labels...")
    samples_dict             = pu.extract_samples(idata)
    mu_bg, z                 = pu.get_bg_residuals(data_dict, samples_dict)
    soft_labels, hard_labels = pu.classify(z, samples_dict, sigma_clean=sigma_clean)

    mean_soft_labels = soft_labels.mean(axis=0)
    std_soft_labels  = soft_labels.std(axis=0)
    mean_hard_labels = (mean_soft_labels > 0.5).astype(int)

    data_dict['hard_labels']     = mean_hard_labels
    data_dict['soft_labels']     = mean_soft_labels
    data_dict['soft_labels_std'] = std_soft_labels

    # ---- Full per-draw soft labels -------------------------------------
    print("\nWriting full per-draw soft labels to a separate HDF5 file...")
    hdf5_utils.write_soft_label_samples(
        h5_path      = hdf5_utils.soft_label_samples_path(run_dir, output_timestring),
        soft_labels  = soft_labels,
        n_chains     = diagnostics['n_chains_after'],
        n_draws      = diagnostics['n_draws_per_chain'],
        pointing     = pointing,
        tv_subband   = tv_subband,
        polarization = polarization,
        data_dict    = data_dict,
        thin         = SOFT_LABEL_THIN,
        source_file  = output_file,
    )

    # ---- Per-night plots ----------------------------------------------
    print("\nMaking per-night prediction plots...")
    for ndx in range(len(data_dict['y_loc'])):
        plu.plot_night(idata, z, data_dict, ndx, mu_bg, pointing, sigma_clean,
                    savefig=True, 
                    savepath=savepath,
                    )

    # ---- Goodness-of-fit metrics --------------------------------------
    print("\nCalculating chi-squared and neg-log-pdf...")
    chi2_median, chi2_lo, chi2_hi = pu.get_chi2(z, sigma_clean, clean_mask=(hard_labels == 0))
    plu.plot_chisq(chi2_median, chi2_lo, chi2_hi, savepath=savepath, savefig=True)

    nll_median, nll_lo, nll_hi = pu.get_neg_logpdf_rfi(z, samples_dict, rfi_mask=(hard_labels == 1))
    plu.plot_nll(nll_median, nll_lo, nll_hi, savepath=savepath, savefig=True)


    # # ---- Residual histograms ------------------------------------------
    print("\nPlotting clean/RFI residual histograms...")
    plu.plot_clean_rfi_histograms(z, data_dict, samples_dict, sigma_clean, savepath=savepath, savefig=True)

    print("\nSummarizing posterior residuals...")
    residual_median, residual_std = pu.get_residual_summary(z)

    print("\nWriting results to central HDF5 file...")
    if half is not None:
        h5_path = os.path.join(run_dir, "processed", f"half{half}_"+hdf5_utils.HDF5_FILENAME)
    else:
        h5_path = os.path.join(run_dir, "processed", hdf5_utils.HDF5_FILENAME)
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    hdf5_utils.write_results(
        h5_path         = h5_path,
        pointing        = pointing,
        tv_subband      = tv_subband,
        polarization    = polarization,
        diagnostics     = diagnostics,
        samples_dict    = samples_dict,
        data_dict       = data_dict,
        chi2            = (chi2_median, chi2_lo, chi2_hi),
        nll             = (nll_median,  nll_lo,  nll_hi),
        residuals       = (residual_median, residual_std),
        rejected_obsids = rejected_obsids,
    )

    print(f"\nDone with: {output_file}\n")


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    parser = cli.build_parser(
        description="Pipeline stage 3: postprocess one fit's chains -- diagnostics, "
                    "labels, goodness-of-fit -- into the shared season results file.",
        n_jobs=len(JOB_MAP),
        epilog="Examples:\n"
               "  python postprocess.py 0\n"
               "  python postprocess.py 0 --run-dir out_chrono_split --half 0\n"
               "  python postprocess.py --n-jobs",
    )
    cli.add_run_dir(parser, default="out",
                    help="run directory to read chains/ from and write processed/ into")
    cli.add_half(parser)
    args = cli.parse_args(parser)

    run_dir = cli.resolve_run_dir(args.run_dir, ABS_DIR)

    L, tv_subband, pointing, polarization = JOB_MAP[args.job_index]

    cli.print_config(
        "Postprocess chains",
        job_index    = args.job_index,
        pointing     = pointing,
        L            = L,
        polarization = polarization,
        tv_subband   = tv_subband,
        half         = args.half,
        run_dir      = run_dir,
    )

    postprocess_chains(L, tv_subband, pointing, polarization, run_dir, half=args.half)
