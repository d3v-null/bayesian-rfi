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


ABS_DIR     = "/oscar/data/jpober/jmduchar/Research/BayesRFI/"
CB_PATH     = os.path.join(os.path.dirname(__file__), "coarse_bands_1D.npy")
CB          = np.load(CB_PATH)
TV_SUBBANDS = ('subTV', 'TV6', 'TV7', 'TV8', 'TV9')

# Raw visibilities in, reduced per-sub-band time series out.
DATA_DIR = "/users/jmduchar/data/shared/big_dataset/"
OUT_DIR  = os.path.join(ABS_DIR, "data/")


def build_initial_uvflag(ins, obs, uvf_save_dir, drop_first_last):
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
    """
    uvf = UVFlag()
    for attr in uvf.__dict__:
        if hasattr(ins, attr):
            setattr(uvf, attr, copy.copy(getattr(ins, attr)))

    # Set mode to flag
    uvf.to_flag()

    # Overwrite flags (just need CB and first-last flags)
    uvf.flag_array = np.zeros_like(uvf.flag_array)
    uvf.flag_array[:,CB,:] = 1

    if drop_first_last:
        uvf.flag_array[0,:,:] = 1
        uvf.flag_array[-1,:,:] = 1
    
    # Label appropriately
    uvf.label=obs

    # Save it out for downstream flag propagation
    uvf.write(os.path.join(uvf_save_dir, f"{obs}_flags.h5"), clobber=True)


def build_ins(obsid, data_dir, uvf_save_dir, drop_first_last=True, verbose=False):
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
        Directory containing raw .fits files.
    uvf_save_dir : str
        Output directory for the UVFlag file.
    drop_first_last : bool, optional
        If True, flag and then drop the first and last time integrations.
        Default True.
    verbose : bool, optional
        If True, print out additional info to stdout.

    Returns
    -------
    INS
        Flagged INS object with edge integrations removed (if drop_first_last).

    Raises
    ------
    ValueError
        If the number of frequency channels in the loaded data is not 768.
    """

    obsfiles = glob.glob(data_dir+f"*{obsid}*fits")
    ss = SS()
    ss.read(obsfiles, diff=True)

    if verbose:
        print(f"OBSID {obsid} successfully read in as an SS object.")

    if ss.Nfreqs != 768:
        raise ValueError(f"Wrong Nfreq value for {obsid}. Expected 768, got {ss.Nfreqs}")

    initial_flags = np.broadcast_to(CB[np.newaxis, :, np.newaxis], (ss.Ntimes, ss.Nfreqs, ss.Npols)).copy()

    if drop_first_last:
        initial_flags[[0, -1], :, :] = True

    initial_ss_flags = np.reshape(
        np.broadcast_to(initial_flags[:, np.newaxis, :, :], (ss.Ntimes, ss.Nbls, ss.Nfreqs, ss.Npols)).copy(),
        (ss.Ntimes * ss.Nbls, ss.Nfreqs, ss.Npols)
    )
    ss.apply_flags(flag_choice="custom", custom=initial_ss_flags)

    ins = INS(ss, spectrum_type = 'cross', order=0)

    if verbose:
        print(f"OBSID {obsid} successfully converted to INS object.")

    shape_dict = {
        'TV6': [1.74e8, 1.81e8],
        'TV7': [1.81e8, 1.88e8],
        'TV8': [1.88e8, 1.95e8],
        'TV9': [1.95e8, 2.02e8],
        }
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

    build_initial_uvflag(ins, obsid, uvf_save_dir, drop_first_last)

    if drop_first_last:
        ins.select(times=np.unique(ins.time_array[1:-1]))
        
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
        Directory containing raw .fits files.
    out_dir : str
        Root output directory. Subdirectories tv_split/{tv_subband} and
        uvflag/ are created here if they don't exist.
    drop_first_last : bool, optional
        If True, flag and drop the first and last time integrations. Default True.
    verbose : bool, optional
        If True, print out additional info to stdout.
    """

    date, pointing = night_pointing_key
    obsids = np.sort(night_pointing_dict[night_pointing_key])

    if verbose:
        print(f"Found {len(obsids)} OBSIDs for night of {date}, pointing {pointing}.")

    uvf_save_dir = os.path.join(out_dir, "uvflag")
    os.makedirs(uvf_save_dir, exist_ok=True)

    ins = None

    for obs in obsids:
        try:
            sub_ins = build_ins(
                obsid           = obs,
                data_dir        = in_dir,
                drop_first_last = drop_first_last,
                uvf_save_dir    = uvf_save_dir,
                verbose         = verbose
                )
        except (TypeError, ValueError):  # some files are corrupted on disk
            continue

        if ins is None:
            ins = sub_ins
        else:
            ins += sub_ins

    if ins is None:
        raise RuntimeError("build_ins failed for all obsids")
        
    del sub_ins

    t_padded = mm_utils.pad_time_array(ins.time_array)

    flags       = ins.metric_array.mask[:,:,0].copy() # SSINS flags are 4 identical copies along pol axis
    flags[:,CB] = 0  # Unmask CB so the 1D avg only considers valid channels
    
    # Iterate over all sub-bands
    for tv_subband in TV_SUBBANDS:
        save_dir = os.path.join(out_dir, f"tv_split/{tv_subband}")
        os.makedirs(save_dir, exist_ok=True)

        freq_mask, freq_mask_flags = get_freq_mask(ins.freq_array, tv_subband)

        flags_subband    = flags[:, freq_mask_flags]
        flags_subband_1d = np.any(flags_subband, axis=-1)

        metric_subband_xx  = ins.metric_array[:, freq_mask, 0]
        metric_subband_xx  = np.where(np.isfinite(metric_subband_xx), metric_subband_xx, np.nan)
        
        metric_subband_yy  = ins.metric_array[:, freq_mask, 1]
        metric_subband_yy  = np.where(np.isfinite(metric_subband_yy), metric_subband_yy, np.nan)
        
        avg_metric_subband_xx = np.nanmean(metric_subband_xx, axis = (1,))
        avg_metric_subband_yy = np.nanmean(metric_subband_yy, axis = (1,))

        np.savez(
            os.path.join(save_dir, f"processed_{obsids[0][:6]}_p{pointing}.npz"),
            xx         = mm_utils.pad_ssins(avg_metric_subband_xx, t_padded, ins.time_array),
            yy         = mm_utils.pad_ssins(avg_metric_subband_yy, t_padded, ins.time_array),
            ssins      = mm_utils.pad_ssins(flags_subband_1d,      t_padded, ins.time_array),
            time_jd    = t_padded,
            obsids     = obsids,
            tv_subband = tv_subband,
            pointing   = pointing,
            date       = date
        )

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
               "  python preprocess.py --n-jobs",
    )
    args = cli.parse_args(parser)

    date, pointing = list(get_season().keys())[args.job_index]

    cli.print_config(
        "Preprocess raw visibilities",
        job_index = args.job_index,
        night     = date,
        pointing  = pointing,
        data_dir  = DATA_DIR,
        out_dir   = OUT_DIR,
    )

    preprocess_single_night(args.job_index)
