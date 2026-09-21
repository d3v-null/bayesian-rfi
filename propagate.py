import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

import glob
import warnings

import cli
import hdf5_utils
import numpy as np
from pyuvdata import UVFlag

warnings.filterwarnings("ignore")


ABS_DIR = "/oscar/data/jpober/jmduchar/Research/BayesRFI/"

TV_BANDS = {
    'subTV': (0,       174e6),
    'TV6':   (174e6,   181e6),
    'TV7':   (181e6,   188e6),
    'TV8':   (188e6,   195e6),
    'TV9':   (195e6,   np.inf),
}

POL_TO_IDX = {
    'xx': 0,
    'yy': 1,
}

TV_SUBBANDS   = ['subTV', 'TV6', 'TV7', 'TV8', 'TV9']
POLARIZATIONS = ['xx', 'yy']

UVFLAG_DIR = os.path.join(ABS_DIR, "data/uvflag/")
H5_PATH    = os.path.join(ABS_DIR, "out/processed/", hdf5_utils.HDF5_FILENAME)

# Number of contiguous chunks the UVFlag file list is split into, one per array task
N_JOBS = 50

# atol precision for numpy isclose
TIME_DECIMALS = 7


def get_freq_mask(uvf, tv_subband):
    """Return boolean mask over freq_array for the given TV subband."""
    lo, hi = TV_BANDS[tv_subband]
    return (uvf.freq_array > lo) & (uvf.freq_array < hi)


def build_label_lookup(df):
    """
    Build a fast time -> {tv_subband: {pol: (soft_label, soft_label_std)}}
    lookup from the season-results dataframe, so per-timestep propagation
    doesn't have to linearly scan the full (multi-million-row) dataframe for
    every UVFlag timestep.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'time_jd', 'tv_subband', 'pol', 'soft_label',
        'soft_label_std' (a subset of the flat dataframe from
        hdf5_utils.read_season_results).

    Returns
    -------
    lookup : dict
        Maps round(time_jd, TIME_DECIMALS) -> {tv_subband: {pol: (soft_label, soft_label_std)}}.
    """
    lookup = {}
    for row in df.itertuples(index=False):
        key = round(row.time_jd, TIME_DECIMALS)
        lookup.setdefault(key, {}).setdefault(row.tv_subband, {})[row.pol] = (
            row.soft_label, row.soft_label_std
        )
    return lookup


def build_rejected_lookup(rejected_obsids):
    """
    Invert hdf5_utils.read_rejected_obsids's per-combo obsid lists into a
    fast obsid -> {tv_subband, ...} lookup for propagate_flags_for_file.

    Parameters
    ----------
    rejected_obsids : dict
        (pointing, tv_subband) -> list of obsid strings, from
        `hdf5_utils.read_rejected_obsids`.

    Returns
    -------
    lookup : dict
        obsid -> set of tv_subband strings rejected for that obsid's night.
        An obsid belongs to exactly one pointing, so pointing isn't needed
        as part of the key.
    """
    lookup = {}
    for (_pointing, tv_subband), obsids in rejected_obsids.items():
        for obsid in obsids:
            lookup.setdefault(obsid, set()).add(tv_subband)
    return lookup


def propagate_flags_for_file(uvfile, lookup, rejected_lookup):
    """
    EXPERIMENTAL
    
    Propagate season-results RFI classifications into one UVFlag file, in
    place, and write the result back to disk in "metric" mode.

    Previously this wrote a boolean hard-label into flag_array. Instead, it
    now writes the continuous soft label (posterior mean RFI responsibility
    p_rfi, in [0, 1]) into `metric_array`, and its Bayesian error bar
    (posterior std of p_rfi) into `weights_array` -- pyuvdata's UVFlag schema
    has no dedicated uncertainty field, but `weights_array` is the only other
    float array of matching shape available in metric mode (see
    https://pyuvdata.readthedocs.io/en/latest/uvflag.html). This is a
    repurposing, not an established pyuvdata convention, so it's documented
    here and in the object's `history`.

    `uvf.to_metric()` is called first, which seeds `metric_array` from the
    object's existing boolean `flag_array` (the coarse-band/edge-time
    pre-flags from `preprocess.py` become metric=1.0, i.e. maximally
    "RFI-like", before classified values overwrite them below).

    For every timestep, looks up the classified (soft_label, soft_label_std)
    for each (TV sub-band, polarization) combination at that time and writes
    it into the corresponding frequency channels (broadcast across the whole
    sub-band, since the mixture model classifies one value per
    frequency-averaged sub-band/timestep). The xx/yy values go onto
    polarization axes 0/1; the two cross-polarization axes (2 = XY, 3 = YX)
    take whichever of xx/yy is more RFI-like (max soft_label), carrying that
    pol's error bar along -- the continuous analogue of the old OR-of-hard-labels.

    If this obsid's night was rejected upstream (excessive SSINS flagging,
    see `mixture_model.reject_nights`), there's no posterior classification
    to look up for the rejected sub-band(s) at all -- so instead of being
    left untouched, every timestep is set to metric=1.0 (treated as fully
    RFI, matching the old force-flag behavior) with weights=0.0, uniformly
    across that sub-band's frequency channels (all 4 polarization axes) for
    the whole file. NOTE: here weights=0.0 means "no posterior data available
    at all" (the night was excluded from the fit), NOT that the uncertainty
    on the metric is actually zero -- don't confuse it with the weights=std
    convention used for classified entries. Timesteps with no matching
    season-results entry for some other reason (e.g. postprocessing hasn't
    run yet for that combo) are left untouched and counted in `n_missing`.

    Parameters
    ----------
    uvfile : str
        Path to a UVFlag `*_flags.h5` file (read AND overwritten in place).
    lookup : dict
        Time -> {tv_subband: {pol: (soft_label, soft_label_std)}} lookup,
        from build_label_lookup.
    rejected_lookup : dict
        obsid -> set of tv_subband strings rejected for that obsid's night,
        from build_rejected_lookup.

    Returns
    -------
    n_missing : int
        Number of timesteps with no matching season-results entry (excluding
        those explained by a rejected sub-band).
    """
    uvf = UVFlag()
    uvf.read(uvfile)
    uvf.to_metric()
    uvf.history += (
        " metric_array = posterior mean soft label (p_rfi); "
        "weights_array = posterior std of the soft label (its Bayesian error "
        "bar), except weights=0 for force-flagged rejected sub-bands, which "
        "means 'no data' rather than 'zero uncertainty'. "
        "See propagate.py."
    )

    obsid = re.sub(r'_flags\.h5$', '', os.path.basename(uvfile))
    rejected_subbands = rejected_lookup.get(obsid, set())

    # ---- Rejected sub-bands: treat as fully RFI, no posterior information ----
    for tv_subband in rejected_subbands:
        freq_mask = get_freq_mask(uvf, tv_subband)
        if freq_mask.any():
            uvf.metric_array[:, freq_mask, :]  = 1.0
            uvf.weights_array[:, freq_mask, :] = 0.0
    if rejected_subbands:
        print(f"  {os.path.basename(uvfile)}: force-flagged rejected sub-band(s) "
              f"{sorted(rejected_subbands)} (excluded from mixture-model fit)")

    n_missing = 0
    for tdx, t in enumerate(uvf.time_array):
        labels_at_t = lookup.get(round(float(t), TIME_DECIMALS))
        if labels_at_t is None:
            n_missing += 1
            continue

        sub_metric  = uvf.metric_array[tdx, :, :]
        sub_weights = uvf.weights_array[tdx, :, :]

        for tv_subband in TV_SUBBANDS:
            if tv_subband in rejected_subbands:
                continue  # already handled above

            subband_labels = labels_at_t.get(tv_subband)
            if subband_labels is None:
                continue

            freq_mask = get_freq_mask(uvf, tv_subband)
            if not freq_mask.any():
                continue

            pol_values = {}
            for pol in POLARIZATIONS:
                if pol not in subband_labels:
                    continue
                soft_label, soft_label_std = subband_labels[pol]
                pol_idx = POL_TO_IDX[pol]
                sub_metric[freq_mask, pol_idx]  = soft_label
                sub_weights[freq_mask, pol_idx] = soft_label_std
                pol_values[pol] = (soft_label, soft_label_std)

            # Cross-polarization axes (XY, YX): take the more RFI-like of
            # xx/yy (max soft_label), carrying its error bar along --
            # the continuous analogue of the old OR-of-hard-labels.
            if pol_values:
                cross_pol = max(pol_values, key=lambda p: pol_values[p][0])
                cross_label, cross_std = pol_values[cross_pol]
                sub_metric[freq_mask, 2]  = cross_label
                sub_weights[freq_mask, 2] = cross_std
                sub_metric[freq_mask, 3]  = cross_label
                sub_weights[freq_mask, 3] = cross_std

        uvf.metric_array[tdx, :, :]  = sub_metric
        uvf.weights_array[tdx, :, :] = sub_weights

    uvf.write(uvfile, clobber=True)
    return n_missing


def propagate_flags(job_index, n_jobs, uvflag_dir=UVFLAG_DIR, h5_path=H5_PATH):
    """
    Propagate season-results RFI classifications into this job's slice of
    UVFlag files.

    The full list of `*_flags.h5` files is split into `n_jobs` contiguous
    chunks (via np.array_split on the sorted file list); this call processes
    only the `job_index`'th chunk. Each task reads/writes a disjoint set of
    files, so this is safe to run as an uncoordinated SLURM array job (unlike
    postprocess.py, no shared-file locking is needed). Chunking
    (rather than one file per task) amortizes the cost of loading and
    indexing the shared season_results.h5 file over many UVFlag files per job.

    Parameters
    ----------
    job_index : int
        Index of this job's chunk, 0 <= job_index < n_jobs.
    n_jobs : int
        Total number of parallel jobs the file list is split across.
    uvflag_dir : str, optional
        Directory containing `*_flags.h5` UVFlag files.
    h5_path : str, optional
        Path to the shared season_results.h5 file.
    """
    all_uvf = sorted(glob.glob(os.path.join(uvflag_dir, "*_flags.h5")))
    if not all_uvf:
        raise FileNotFoundError(f"No UVFlag files found in {uvflag_dir}")

    my_files = np.array_split(all_uvf, n_jobs)[job_index].tolist()
    print(f"Job {job_index}/{n_jobs}: processing {len(my_files)} of {len(all_uvf)} UVFlag files")

    print("Loading season results and building time lookup...")
    df = hdf5_utils.read_season_results(h5_path)
    lookup = build_label_lookup(df[['time_jd', 'tv_subband', 'pol', 'soft_label', 'soft_label_std']])
    del df

    print("Loading rejected nights and building obsid lookup...")
    rejected_obsids = hdf5_utils.read_rejected_obsids(h5_path)
    rejected_lookup = build_rejected_lookup(rejected_obsids)

    for uvfile in my_files:
        n_missing = propagate_flags_for_file(uvfile, lookup, rejected_lookup)
        print(f"  {os.path.basename(uvfile)}: {n_missing} timestep(s) with no matching season-results entry")

    print(f"Done with job {job_index}/{n_jobs}")


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    parser = cli.build_parser(
        description="Pipeline stage 5: write this task's chunk of UVFlag files with "
                    "the soft labels and error bars from a run's season results.",
        n_jobs=N_JOBS,
        epilog="Examples:\n"
               "  python propagate.py 0\n"
               "  python propagate.py 0 --run-dir out_chrono_split\n"
               "  python propagate.py --n-jobs",
    )
    cli.add_run_dir(parser, default="out",
                    help="run directory to read processed/season_results.h5 from")
    args = cli.parse_args(parser)

    run_dir = cli.resolve_run_dir(args.run_dir, ABS_DIR)
    h5_path = os.path.join(run_dir, "processed", hdf5_utils.HDF5_FILENAME)

    cli.print_config(
        "Propagate flags",
        job_index  = args.job_index,
        n_jobs     = N_JOBS,
        run_dir    = run_dir,
        h5_path    = h5_path,
        uvflag_dir = UVFLAG_DIR,
    )

    propagate_flags(args.job_index, N_JOBS, h5_path=h5_path)
