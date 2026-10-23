import copy
import functools
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

import glob
import warnings

import cli
import mm_utils
import numpy as np
from pyuvdata import UVFlag
from SSINS import INS, MF, SS

warnings.filterwarnings("ignore")


ABS_DIR     = os.environ.get("BAYESRFI_ABS_DIR", "/oscar/data/jpober/jmduchar/Research/BayesRFI/")
TV_SUBBANDS = ('subTV', 'TV6', 'TV7', 'TV8', 'TV9')

# MWA coarse-channel layout used to build the coarse-band flag mask.
N_COARSE        = 24      # coarse channels in a full legacy observation
COARSE_BW_HZ    = 1.28e6  # coarse channel width
# width flagged at each coarse-channel edge (legacy-PFB aliasing). MWAX Phase III (oversampled PFB) has no
# aliased edges and no DC spike: set BAYESRFI_EDGE_FLAG_HZ=0 BAYESRFI_FLAG_DC=0 for that data.
EDGE_FLAG_HZ    = float(os.environ.get('BAYESRFI_EDGE_FLAG_HZ', 80e3))
FLAG_DC         = os.environ.get('BAYESRFI_FLAG_DC', '1') != '0'

# Raw visibilities in, reduced per-sub-band time series out.
DATA_DIR = os.environ.get("BAYESRFI_DATA_DIR", "/users/jmduchar/data/shared/big_dataset/")
OUT_DIR  = os.path.join(ABS_DIR, "data/")

# How each obsid's visibilities are found in DATA_DIR (`--input`). "gpubox" is the
# raw correlator FITS + metafits set the 2014 analysis used; "uvfits" and "ms" are
# preprocessed files (e.g. from Birli), which is the only way in for MWAX data
# with correlator-side delay tracking (DELAYMOD=FULLTRACK), which pyuvdata 3.x
# refuses to read raw. "ins" is an INS already computed from the visibilities, e.g. by
# `hyperdrive bmetrics --ins`, in the format of SSINS's `INS.write(output_type="data")`.
INPUT_GLOBS = {
    'gpubox': "*{obsid}*fits",
    'uvfits': "*{obsid}*.uvfits",
    'ms':     "*{obsid}*.ms",
    'ins':    "{obsid}_SSINS_data.h5",
}

# DTV match-filter shapes (Hz); see Table 1 of the paper.
SHAPE_DICT = {
    'TV6': [1.74e8, 1.81e8],
    'TV7': [1.81e8, 1.88e8],
    'TV8': [1.88e8, 1.95e8],
    'TV9': [1.95e8, 2.02e8],
}


def coarse_band_mask(freq_array, n_coarse=None, coarse_bw=COARSE_BW_HZ, edge_flag=EDGE_FLAG_HZ):
    """
    Boolean mask of the fine channels to flag in every MWA coarse channel: the
    `edge_flag` Hz at each edge (aliased by the polyphase filterbank) and the
    centre (DC) channel. Derived from the frequency axis so it works for any
    fine-channel width (10/20/40 kHz, ...), not just the 768-channel 40 kHz layout.

    Parameters
    ----------
    freq_array : ndarray
        Fine-channel centre frequencies in Hz, ascending.
    n_coarse : int, optional
        Number of coarse channels the frequency axis spans. Inferred from its
        total bandwidth if omitted, so a single coarse channel (e.g. one MWAX
        `_ch###` file) works as well as the full 24-channel band.

    Returns
    -------
    ndarray of bool, shape (Nfreqs,)
        True for channels to flag.
    """
    freq_array = np.asarray(freq_array).ravel()
    nfreqs = freq_array.size
    if n_coarse is None:
        df = float(np.median(np.diff(freq_array)))
        n_coarse = max(1, round(nfreqs * df / coarse_bw))
    if nfreqs % n_coarse:
        raise ValueError(f"Nfreqs={nfreqs} is not a multiple of {n_coarse} coarse channels")
    n_fine = nfreqs // n_coarse
    df     = float(np.median(np.diff(freq_array)))
    if not np.isclose(n_fine * df, coarse_bw, rtol=1e-3):
        raise ValueError(f"{n_fine} fine channels x {df/1e3:.1f} kHz != {coarse_bw/1e6} MHz coarse channel")
    n_edge = int(round(edge_flag / df))
    cb = np.zeros(nfreqs, dtype=bool)
    for c in range(n_coarse):
        if n_edge:
            cb[c * n_fine : c * n_fine + n_edge]          = True
            cb[(c + 1) * n_fine - n_edge : (c + 1) * n_fine] = True
        if FLAG_DC:
            cb[c * n_fine + n_fine // 2]              = True
    return cb


def build_initial_uvflag(ins, obs, uvf_save_dir, drop_first_last, cb):
    """
    Build and save a UVFlag seeded with coarse-band and edge-time flags.

    Constructs a UVFlag from an INS object's metadata, flags coarse-band
    channels and (optionally) the first/last time integrations, then writes
    the result to disk for downstream flag propagation.

    Parameters
    ----------
    ins : INS
        Source INS object
    obs : str
        MWA OBSID, used as the UVFlag label and in the output filename.
    uvf_save_dir : str
        Directory where the output .h5 file will be written.
    drop_first_last : bool
        If True, flag the first and last time integrations.
    cb : ndarray of bool
        Coarse-band channel mask from `coarse_band_mask`.
    """
    uvf = UVFlag()
    for attr in uvf.__dict__:
        if hasattr(ins, attr):
            setattr(uvf, attr, copy.copy(getattr(ins, attr)))

    # Set mode to flag
    uvf.to_flag()

    # Overwrite flags (just need CB and first-last flags)
    uvf.flag_array = np.zeros_like(uvf.flag_array)
    uvf.flag_array[:,cb,:] = 1

    if drop_first_last:
        uvf.flag_array[0,:,:] = 1
        uvf.flag_array[-1,:,:] = 1
    
    # Label appropriately
    uvf.label=obs

    # Save it out for downstream flag propagation
    uvf.write(os.path.join(uvf_save_dir, f"{obs}_flags.h5"), clobber=True)


def match_filter_shapes(freq_array, shape_dict=SHAPE_DICT):
    """
    The DTV shapes that overlap the observed band.

    SSINS turns a shape lying wholly outside the band into a spurious one-channel
    shape at the band edge, so with a single coarse channel only the sub-band(s)
    it actually sits in are kept.
    """
    lo, hi = float(np.min(freq_array)), float(np.max(freq_array))
    return {name: band for name, band in shape_dict.items() if band[0] < hi and band[1] > lo}


def _ins_from_visibilities(obsfiles, drop_first_last):
    """
    SSINS INS of the cross-correlations of one observation's visibility files,
    with the coarse-band and (optionally) first/last-time flags applied first.
    Returns (ins, number of cross-correlation baselines averaged).
    """
    ss = SS()
    ss.read(obsfiles if len(obsfiles) > 1 else obsfiles[0], diff=True)

    cb = coarse_band_mask(ss.freq_array)

    initial_flags = np.broadcast_to(cb[np.newaxis, :, np.newaxis], (ss.Ntimes, ss.Nfreqs, ss.Npols)).copy()

    if drop_first_last:
        initial_flags[[0, -1], :, :] = True

    initial_ss_flags = np.reshape(
        np.broadcast_to(initial_flags[:, np.newaxis, :, :], (ss.Ntimes, ss.Nbls, ss.Nfreqs, ss.Npols)).copy(),
        (ss.Ntimes * ss.Nbls, ss.Nfreqs, ss.Npols)
    )
    ss.apply_flags(flag_choice="custom", custom=initial_ss_flags)

    ins = INS(ss, spectrum_type = 'cross', order=0)

    # spectrum_type='cross' averages cross-correlations only, so autos don't count.
    cross = ss.ant_1_array != ss.ant_2_array
    return ins, int(np.unique(ss.baseline_array[cross]).size)


def _read_precomputed_ins(path, drop_first_last):
    """
    Read a cross-correlation INS written in SSINS's `INS.write(output_type="data")`
    format (e.g. `hyperdrive bmetrics --ins`) and apply the same coarse-band and
    first/last-time flags `_ins_from_visibilities` applies to the visibilities.
    Its weights are the number of baselines averaged per (time, channel).
    Returns (ins, number of baselines averaged).
    """
    ins = INS(path)
    if "spectrum_type:cross" not in ins.history:
        raise ValueError(f"{path} is not a cross-correlation INS")

    cb = coarse_band_mask(ins.freq_array)
    ins.metric_array[:, cb, :] = np.ma.masked
    if drop_first_last:
        ins.metric_array[[0, -1], :, :] = np.ma.masked
    ins.metric_ms = ins.mean_subtract()

    return ins, int(ins.weights_array.max())


def build_ins(obsid, data_dir, uvf_save_dir, drop_first_last=True, verbose=False, input_format='gpubox'):
    """
    Load a visibility file, apply initial flags, run the default SSINS match filter,
    and return the flagged INS object.

    Applies coarse-band and (optionally) edge-time flags before computing the
    incoherent noise spectrum. Runs a match filter tuned to DTV shapes and
    narrowband/streak RFI. Saves a UVFlag file seeded with the initial flags
    for downstream propagation, then trims edge integrations from the returned
    INS if drop_first_last is True.

    Parameters
    ----------
    obsid : str
        MWA OBSID used to glob for input .fits files in data_dir.
    data_dir : str
        Directory containing the input visibility files.
    uvf_save_dir : str
        Output directory for the UVFlag file.
    drop_first_last : bool, optional
        If True, flag and then drop the first and last time integrations.
        Default True.
    verbose : bool, optional
        If True, print out additional info to stdout.
    input_format : str, optional
        Key of INPUT_GLOBS: 'gpubox' (raw correlator FITS + metafits, default),
        'uvfits' or 'ms' (preprocessed, e.g. by Birli), or 'ins' (a precomputed
        cross-correlation INS, e.g. from `hyperdrive bmetrics --ins`).

    Returns
    -------
    INS
        Flagged INS object with edge integrations removed (if drop_first_last).

    Raises
    ------
    ValueError
        If the frequency axis does not match the MWA coarse-channel layout
        (see `coarse_band_mask`).
    """

    obsfiles = sorted(glob.glob(os.path.join(data_dir, INPUT_GLOBS[input_format].format(obsid=obsid))))
    if not obsfiles:
        raise ValueError(f"no {input_format} input for OBSID {obsid} in {data_dir}")
    if input_format == 'ins':
        ins, nbls = _read_precomputed_ins(obsfiles[0], drop_first_last)
    else:
        ins, nbls = _ins_from_visibilities(obsfiles, drop_first_last)

    if verbose:
        print(f"OBSID {obsid} successfully converted to INS object.")

    cb = coarse_band_mask(ins.freq_array)

    shape_dict = match_filter_shapes(ins.freq_array)
    sig_thresh = {shape: 5 for shape in shape_dict}
    sig_thresh["narrow"] = 5
    sig_thresh["streak"] = 10
    mf = MF(
        ins.freq_array,
        sig_thresh,
        shape_dict=shape_dict,
        tb_aggro=0.4,
        broadcast_streak=True,
    )
    mf.apply_match_test(
        ins, event_record=True, time_broadcast=False, freq_broadcast=False
    )

    build_initial_uvflag(ins, obsid, uvf_save_dir, drop_first_last, cb)

    if drop_first_last:
        ins.select(times=np.unique(ins.time_array[1:-1]))

    # INS (a waterfall UVFlag) has Nbls=None; remember how many baselines went
    # into the incoherent average for the thermal-noise estimate downstream.
    ins.nbls_averaged = nbls

    return ins


def _ins_cache_paths(ins_dir, obs):
    """Per-obsid on-disk INS cache: (data file, mask file, nbls sidecar)."""
    prefix = os.path.join(ins_dir, str(obs))
    return prefix + "_SSINS_data.h5", prefix + "_SSINS_mask.h5", prefix + "_nbls.txt"


def _build_ins_safe(obs, build_kwargs, ins_dir=None):
    """
    build_ins wrapper that turns a corrupted/unreadable obsid into None (with a
    warning). When `ins_dir` is given the INS is written there (data + mask) and
    the cache paths are returned instead of the object: INS objects cannot be
    pickled back from a worker process (SSINS `_data_params` breaks in
    `__setstate__`), and a cached INS lets a night be resumed after a crash.
    """
    if ins_dir is not None:
        data_path, mask_path, nbls_path = _ins_cache_paths(ins_dir, obs)
        if os.path.exists(data_path) and os.path.exists(mask_path) and os.path.exists(nbls_path):
            print(f"OBSID {obs}: using cached INS in {ins_dir}")
            return (data_path, mask_path, nbls_path)
    try:
        ins = build_ins(obsid=obs, **build_kwargs)
    except (TypeError, ValueError) as e:  # some files are corrupted on disk
        print(f"WARNING: skipping OBSID {obs}: {type(e).__name__}: {e}")
        return None
    if ins_dir is None:
        return ins
    os.makedirs(ins_dir, exist_ok=True)
    ins.write(os.path.join(ins_dir, str(obs)), output_type="data", clobber=True)
    ins.write(os.path.join(ins_dir, str(obs)), output_type="mask", clobber=True)
    with open(nbls_path, "w") as f:
        f.write(f"{ins.nbls_averaged}\n")
    return (data_path, mask_path, nbls_path)


def _load_cached_ins(paths):
    """Reload an INS written by `_build_ins_safe(..., ins_dir=...)`."""
    data_path, mask_path, nbls_path = paths
    ins = INS(data_path, mask_file=mask_path)
    with open(nbls_path) as f:
        ins.nbls_averaged = int(f.read().strip())
    return ins


def get_freq_mask(freq_array, tv_subband):
    """
    Return boolean frequency masks for a DTV sub-band.

    Parameters
    ----------
    freq_array : ndarray
        Frequency channels in Hz.
    tv_subband : str
        Sub-band name. One of: 'subTV', 'TV6', 'TV7', 'TV8', 'TV9'.

    Returns
    -------
    freq_mask : ndarray of bool
        True for channels within the sub-band. Used for metric averaging.
    flag_mask : ndarray of bool
        True for channels within the sub-band, inset by 0.5 MHz on each edge
        to avoid edge artefacts when extracting flags.

    Raises
    ------
    ValueError
        If tv_subband is not recognised.
    """
    BANDS = {
        'subTV': (None,  174e6),
        'TV6':   (174e6, 181e6),
        'TV7':   (181e6, 188e6),
        'TV8':   (188e6, 195e6),
        'TV9':   (195e6, None),
    }

    if tv_subband not in BANDS:
        raise ValueError(f"Unrecognized tv_subband: {tv_subband!r}. Allowed: {list(BANDS)}")

    lo, hi = BANDS[tv_subband]
    inset = 0.5e6

    freq_mask = (freq_array > lo if lo else True) & (freq_array < hi if hi else True)
    flag_mask = (freq_array > lo + inset if lo else True) & (freq_array < hi - inset if hi else True)

    return freq_mask, flag_mask


def process_night_pointing(
    night_pointing_key, 
    night_pointing_dict, 
    in_dir,
    out_dir,
    drop_first_last=True,
    verbose=False,
    input_format='gpubox',
    ):
    """
    Process all observations for a single night/pointing and save per-subband
    metrics and flags for every TV sub-band.

    Builds and concatenates INS objects for each obsid in the pointing (once,
    shared across all sub-bands), then for each sub-band in TV_SUBBANDS
    computes frequency-averaged XX/YY metrics and a 1D flag timeseries, pads
    them onto a uniform time grid, and saves to a .npz file.

    Output files are written to:
        {out_dir}/tv_split/{tv_subband}/processed_{obsn}_p{pointing}.npz
        {out_dir}/uvflag/{obsid}_flags.h5   (one per obsid)

    Parameters
    ----------
    night_pointing_key : tuple of (str, str)
        (date, pointing) key into night_pointing_dict.
    night_pointing_dict : dict
        Mapping from (date, pointing) keys to arrays of obsids.
    in_dir : str
        Directory containing the input visibility files.
    out_dir : str
        Root output directory. Subdirectories tv_split/{tv_subband} and
        uvflag/ are created here if they don't exist.
    drop_first_last : bool, optional
        If True, flag and drop the first and last time integrations. Default True.
    verbose : bool, optional
        If True, print out additional info to stdout.
    input_format : str, optional
        Passed through to `build_ins`.

    Sub-bands with no channels in the data (e.g. all but one when the input is a
    single coarse channel) are skipped rather than written as all-NaN series.
    """

    date, pointing = night_pointing_key
    obsids = np.sort(night_pointing_dict[night_pointing_key])

    if verbose:
        print(f"Found {len(obsids)} OBSIDs for night of {date}, pointing {pointing}.")

    uvf_save_dir = os.path.join(out_dir, "uvflag")
    os.makedirs(uvf_save_dir, exist_ok=True)

    ins = None
    nbls_per_obs = []

    # Reading one 2-minute MWA observation takes several minutes and ~30 GB,
    # so allow the per-obsid work to run in parallel (BAYESRFI_NPROC, default 1).
    nproc = int(os.environ.get("BAYESRFI_NPROC", "1"))
    build_kwargs = dict(data_dir=in_dir, drop_first_last=drop_first_last,
                        uvf_save_dir=uvf_save_dir, verbose=verbose,
                        input_format=input_format)
    ins_dir = os.path.join(out_dir, "ins")  # per-obsid INS cache (also makes reruns cheap)
    if nproc > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=nproc) as pool:
            futures = [pool.submit(_build_ins_safe, obs, build_kwargs, ins_dir) for obs in obsids]
            results = [f.result() for f in futures]
    else:
        results = [_build_ins_safe(obs, build_kwargs, ins_dir) for obs in obsids]

    for obs, sub_ins in zip(obsids, results):
        if sub_ins is None:
            continue
        if isinstance(sub_ins, tuple):
            sub_ins = _load_cached_ins(sub_ins)
        nbls_per_obs.append(sub_ins.nbls_averaged)
        if ins is None:
            ins = sub_ins
        else:
            ins += sub_ins

    if ins is None:
        raise RuntimeError("build_ins failed for all obsids")

    del sub_ins

    t_padded = mm_utils.pad_time_array(ins.time_array)

    # Number of baselines actually averaged into the INS (pyuvdata drops
    # metafits-flagged tiles by default, so this is < Ntiles*(Ntiles-1)/2).
    nbls = int(np.median(nbls_per_obs))
    if len(set(nbls_per_obs)) > 1:
        print(f"WARNING: Nbls varies across obsids ({min(nbls_per_obs)}-{max(nbls_per_obs)}); using median {nbls}")
    n_channels = {}

    CB          = coarse_band_mask(ins.freq_array)
    flags       = ins.metric_array.mask[:,:,0].copy() # SSINS flags are 4 identical copies along pol axis
    flags[:,CB] = 0  # Unmask CB so the 1D avg only considers valid channels
    
    # Iterate over all sub-bands
    for tv_subband in TV_SUBBANDS:
        save_dir = os.path.join(out_dir, f"tv_split/{tv_subband}")
        os.makedirs(save_dir, exist_ok=True)

        freq_mask, freq_mask_flags = get_freq_mask(ins.freq_array, tv_subband)
        if not (freq_mask & ~CB).any():
            if verbose:
                print(f"No unflagged channels in {tv_subband}; skipping it.")
            continue

        flags_subband    = flags[:, freq_mask_flags]
        flags_subband_1d = np.any(flags_subband, axis=-1)

        metric_subband_xx  = ins.metric_array[:, freq_mask, 0]
        metric_subband_xx  = np.where(np.isfinite(metric_subband_xx), metric_subband_xx, np.nan)
        
        metric_subband_yy  = ins.metric_array[:, freq_mask, 1]
        metric_subband_yy  = np.where(np.isfinite(metric_subband_yy), metric_subband_yy, np.nan)
        
        # XY (pyuvdata pol index 2, -7) for cross-pol fits (BAYESRFI_POLS=xx,yy,xy)
        metric_subband_xy  = ins.metric_array[:, freq_mask, 2] if ins.metric_array.shape[-1] > 2 else np.full_like(metric_subband_xx, np.nan)
        metric_subband_xy  = np.where(np.isfinite(metric_subband_xy), metric_subband_xy, np.nan)

        avg_metric_subband_xx = np.nanmean(metric_subband_xx, axis = (1,))
        avg_metric_subband_yy = np.nanmean(metric_subband_yy, axis = (1,))
        avg_metric_subband_xy = np.nanmean(metric_subband_xy, axis = (1,))

        np.savez(
            os.path.join(save_dir, f"processed_{obsids[0][:6]}_p{pointing}.npz"),
            xx         = mm_utils.pad_ssins(avg_metric_subband_xx, t_padded, ins.time_array),
            yy         = mm_utils.pad_ssins(avg_metric_subband_yy, t_padded, ins.time_array),
            xy         = mm_utils.pad_ssins(avg_metric_subband_xy, t_padded, ins.time_array),
            ssins      = mm_utils.pad_ssins(flags_subband_1d,      t_padded, ins.time_array),
            time_jd    = t_padded,
            obsids     = obsids,
            tv_subband = tv_subband,
            pointing   = pointing,
            date       = date,
            nbls       = nbls,
            nchan      = int((freq_mask & ~CB).sum()),
        )
        n_channels[tv_subband] = int((freq_mask & ~CB).sum())

    # Channels averaged per sub-band, consumed by mixture_model.calculate_sigma_clean
    import yaml
    with open(os.path.join(out_dir, "N_channels.yaml"), "w") as f:
        yaml.safe_dump(n_channels, f)

    if verbose:
        print(f"Finished processing night of {date}, pointing {pointing}.")


@functools.cache
def get_season(datapath=DATA_DIR):
    """
    The season's (date, pointing) -> [obsid, ...] grouping, cached.

    This is the job grid for this stage: one task per (night, pointing) key,
    in dict insertion order. Cached because building it opens every metafits
    header in `datapath`, and both the job-grid size and the night lookup
    itself need it within a single run.
    """
    return mm_utils.sort_season(datapath)


def get_n_jobs(datapath=DATA_DIR):
    """Number of tasks in this stage's job grid (one per night/pointing)."""
    return len(get_season(datapath))


def preprocess_single_night(
    night_idx,
    datapath=DATA_DIR,
    out_dir=OUT_DIR,
    input_format='gpubox',
    ):
    """
    Look up the night_idx'th (date, pointing) in the season and process it.

    Parameters
    ----------
    night_idx : int
        Index into the sorted night/pointing dict returned by mm_utils.sort_season.
    datapath : str, optional
        Directory containing raw .fits files.
    out_dir : str, optional
        Root output directory, passed through to process_night_pointing.
    input_format : str, optional
        Passed through to process_night_pointing (see INPUT_GLOBS).
    """
    night_pointing_dict = get_season(datapath)
    key = list(night_pointing_dict.keys())[night_idx]

    process_night_pointing(
        night_pointing_key  = key,
        night_pointing_dict = night_pointing_dict,
        in_dir              = datapath,
        out_dir             = out_dir,
        drop_first_last     = True,
        verbose             = True,
        input_format        = input_format,
    )


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    parser = cli.build_parser(
        description="Pipeline stage 1: reduce one night/pointing of raw MWA "
                    "visibilities with SSINS into per-sub-band time series.",
        n_jobs=get_n_jobs(),
        epilog="Examples:\n"
               "  python preprocess.py 0\n"
               "  python preprocess.py 0 --input uvfits   # Birli output, e.g. one MWAX coarse channel\n"
               "  python preprocess.py --n-jobs",
    )
    parser.add_argument(
        "--input",
        choices=tuple(INPUT_GLOBS),
        default="gpubox",
        dest="input_format",
        help="visibility files to read per obsid from BAYESRFI_DATA_DIR: raw "
             "correlator FITS + metafits ('gpubox', default), or preprocessed "
             "'uvfits' / 'ms' (e.g. Birli output), or a precomputed "
             "'ins' ({obsid}_SSINS_data.h5, e.g. from hyperdrive bmetrics --ins). "
             "A .metafits per obsid is always needed to group obsids by night and pointing",
    )
    args = cli.parse_args(parser)

    date, pointing = list(get_season().keys())[args.job_index]

    cli.print_config(
        "Preprocess raw visibilities",
        job_index = args.job_index,
        night     = date,
        pointing  = pointing,
        input     = args.input_format,
        data_dir  = DATA_DIR,
        out_dir   = OUT_DIR,
    )

    preprocess_single_night(args.job_index, input_format=args.input_format)
