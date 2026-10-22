import contextlib
import fcntl
import os
import time

import h5py
import numpy as np
import pandas as pd

HDF5_FILENAME = "season_results.h5"

TV_SUBBANDS   = ['subTV', 'TV6', 'TV7', 'TV8', 'TV9']
POINTINGS     = [0, 1, 2, 3, 4]
POLARIZATIONS = ['xx', 'yy']

# Pointing label of a `--pool-pointings` fit; must match mixture_model.POOLED_POINTING
POOLED_POINTING = "all"


def _night_key(night_id):
    """A night id as stored: int for a plain night, str ('144100_p2') for a pooled-fit block."""
    return int(night_id) if str(night_id).isdigit() else str(night_id)

@contextlib.contextmanager
def locked_hdf5(path, mode='a', retries=20, wait=3.0):
    """
    Open an HDF5 file with an exclusive file lock to allow safe
    concurrent writes from parallel SLURM processes.

    Parameters
    ----------
    path    : str   Path to the HDF5 file.
    mode    : str   h5py open mode. Default 'a' (read/write, create if needed).
    retries : int   Number of times to retry acquiring the lock.
    wait    : float Seconds to wait between retries.
    """
    lock_path = path + ".lock"
    for attempt in range(retries):
        try:
            lock_fd = open(lock_path, 'w')
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            lock_fd.close()
            print(f"  HDF5 locked, retrying ({attempt+1}/{retries})...")
            time.sleep(wait)
    else:
        raise TimeoutError(f"Could not acquire lock on {path} after {retries} retries.")

    try:
        with h5py.File(path, mode) as f:
            yield f
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def write_results(h5_path, pointing, tv_subband, polarization,
                  diagnostics, samples_dict, data_dict, chi2, nll, residuals,
                  rejected_obsids):
    """
    Write all results for one (pointing, tv_subband, polarization) combination
    into the central HDF5 file.

    HDF5 structure:
        /{pointing}/{tv_subband}/
            rejected_obsids  (variable-length list of obsid strings)
            {polarization}/
                chain_diagnostics/
                    rhat, ess, n_chains, n_stuck_chains, stuck_chains
                posterior_summary/
                    w_clean/    median, hdi_94_lo, hdi_94_hi, hdi_50_lo, hdi_50_hi
                    loc_rfi/    ...
                    scale_rfi/  ...
                    chisq_per_night/   median, hdi_94_lo, hdi_94_hi, night_ids
                    rfi_nll_per_night/ median, hdi_94_lo, hdi_94_hi, night_ids
                labels/
                    soft_labels       (N_nights, T_max)
                    soft_labels_std   (N_nights, T_max)
                    hard_labels       (N_nights, T_max)
                    residuals_median  (N_nights, T_max)
                    residuals_std     (N_nights, T_max)
                    time_stamps       (N_nights, T_max)
                    night_ids         (N_nights,)
                    lengths           (N_nights,)

    Parameters
    ----------
    residuals : tuple of (residual_median, residual_std)
        Posterior median and std of the background-subtracted residual at
        every (night, timestep), collapsed over posterior draws -- i.e. the
        reconstructed data itself, not just labels/metrics derived from it.
        Each array has shape (N_nights, T_max). See
        `postprocessing_utils.get_residual_summary`.
    rejected_obsids : dict
        Night ID -> list of full MWA obsid strings, for nights excluded from
        this (pointing, tv_subband) combo for excessive SSINS flagging (see
        `mixture_model.reject_nights`). Rejection doesn't depend on
        polarization, so this is written once per (pointing, tv_subband),
        shared by both the xx and yy jobs, rather than duplicated under each
        polarization's group. Consumed by `propagate.py` via
        `read_rejected_obsids`, since rejected nights have no posterior
        classification to look up and should instead be flagged outright.
    """
    chi2_median, chi2_lo, chi2_hi = chi2
    nll_median,  nll_lo,  nll_hi  = nll
    residual_median, residual_std = residuals

    night_ids       = data_dict['night_ids']
    soft_labels     = np.array(data_dict['soft_labels'])
    soft_labels_std = np.array(data_dict['soft_labels_std'])
    hard_labels     = np.array(data_dict['hard_labels'])
    times           = np.array(data_dict['times'])
    lengths         = np.array(data_dict['lengths'])

    combo_root = f"/{pointing}/{tv_subband}"
    root       = f"{combo_root}/{polarization}"

    flat_rejected_obsids = sorted({obs for obsids in rejected_obsids.values() for obs in obsids})

    with locked_hdf5(h5_path, mode='a') as f:

        # ---- rejected obsids (shared across polarizations) ----------------
        combo_grp = f.require_group(combo_root)
        _write_dataset(combo_grp, 'rejected_obsids',
                       np.array(flat_rejected_obsids, dtype=h5py.special_dtype(vlen=str)))


        # ---- chain_diagnostics ----------------------------------------
        cd = f.require_group(f"{root}/chain_diagnostics")
        _write_scalar(cd, 'n_chains',       diagnostics['n_chains_after'])
        _write_scalar(cd, 'n_stuck_chains', diagnostics['n_stuck_chains'])
        _write_scalar(cd, 'max_rhat',       diagnostics['max_rhat_after'])
        _write_scalar(cd, 'median_rhat',    diagnostics['median_rhat_after'])
        _write_scalar(cd, 'mean_ess_bulk',  diagnostics['mean_ess_bulk_after'])
        _write_scalar(cd, 'mean_ess_tail',  diagnostics['mean_ess_tail_after'])
        _write_scalar(cd, 'n_divergences',  diagnostics['n_divergences'] or 0)
        _write_dataset(cd, 'stuck_chains',
                       np.array(diagnostics['stuck_chains'], dtype=int))

        # ---- posterior_summary/scalar params --------------------------
        ps = f.require_group(f"{root}/posterior_summary")
        for key in ['w_clean', 'loc_rfi', 'scale_rfi']:
            from postprocessing_utils import posterior_summary
            median, hdi_outer, hdi_inner = posterior_summary(samples_dict, key)
            grp = ps.require_group(key)
            _write_scalar(grp, 'median',    median)
            _write_scalar(grp, 'hdi_94_lo', hdi_outer[0])
            _write_scalar(grp, 'hdi_94_hi', hdi_outer[1])
            _write_scalar(grp, 'hdi_50_lo', hdi_inner[0])
            _write_scalar(grp, 'hdi_50_hi', hdi_inner[1])

        # ---- posterior_summary/chisq_per_night ------------------------
        cg = ps.require_group('chisq_per_night')
        _write_dataset(cg, 'median',    chi2_median)
        _write_dataset(cg, 'hdi_94_lo', chi2_lo)
        _write_dataset(cg, 'hdi_94_hi', chi2_hi)
        _write_dataset(cg, 'night_ids',
                       np.array(night_ids, dtype=h5py.special_dtype(vlen=str)))

        # ---- posterior_summary/rfi_nll_per_night ----------------------
        ng = ps.require_group('rfi_nll_per_night')
        _write_dataset(ng, 'median',    nll_median)
        _write_dataset(ng, 'hdi_94_lo', nll_lo)
        _write_dataset(ng, 'hdi_94_hi', nll_hi)
        _write_dataset(ng, 'night_ids',
                       np.array(night_ids, dtype=h5py.special_dtype(vlen=str)))

        # ---- labels ---------------------------------------------------
        lg = f.require_group(f"{root}/labels")
        _write_dataset(lg, 'soft_labels',      soft_labels)
        _write_dataset(lg, 'soft_labels_std',  soft_labels_std)
        _write_dataset(lg, 'hard_labels',      hard_labels)
        _write_dataset(lg, 'residuals_median', np.asarray(residual_median))
        _write_dataset(lg, 'residuals_std',    np.asarray(residual_std))
        _write_dataset(lg, 'time_stamps',      times)
        _write_dataset(lg, 'lengths',          lengths)
        _write_dataset(lg, 'night_ids',
                       np.array(night_ids, dtype=h5py.special_dtype(vlen=str)))

    print(f"  Written to HDF5: {root}")


SOFT_LABEL_SAMPLES_SUBDIR = "soft_label_samples"


def soft_label_samples_path(run_dir, output_timestring):
    """
    Path of the per-run HDF5 file holding the full (chain, draw, night, time)
    soft-label array for one postprocessing run.

    Kept out of the shared `season_results.h5` (which only stores the posterior
    mean/std collapsed over draws) because the full array is O(10 GB) per
    (pointing, tv_subband, polarization) combo and would blow up a file that
    every downstream stage opens.

    Parameters
    ----------
    run_dir           : str  Run directory tree (same one stage 3 writes into).
    output_timestring : str  Chains-file stem, e.g. 'TV7_p0_polxx_20260525_114956'
                        (already encodes tv_subband, pointing, polarization,
                        optional half tag, and fit timestamp), so the filename
                        is unique per run.
    """
    return os.path.join(run_dir, "processed", SOFT_LABEL_SAMPLES_SUBDIR,
                        f"soft_label_samples_{output_timestring}.h5")


def write_soft_label_samples(h5_path, soft_labels, n_chains, n_draws,
                             pointing, tv_subband, polarization, data_dict,
                             thin=1, source_file=None, compression='gzip'):
    """
    Write the complete per-draw soft labels (RFI responsibilities) to their own
    HDF5 file, unstacked back into (chain, draw, night, time).

    Unlike the shared season-results store, this file is written by exactly one
    SLURM task and read by nobody concurrently, so it is opened with plain
    `h5py.File` rather than `locked_hdf5`.

    The array is written one chain at a time so the full (S, N, T) block is
    never copied off-device in one piece.

    HDF5 structure:
        /soft_labels  (n_chains, n_draws_kept, N_nights, T_max)  float32
        /night_ids    (N_nights,)              vlen str
        /time_stamps  (N_nights, T_max)
        /lengths      (N_nights,)
        /mask         (N_nights, T_max)        bool, valid (unpadded) samples
        attrs: pointing, tv_subband, polarization, n_chains, n_draws,
               n_draws_kept, thin, source_file

    Parameters
    ----------
    h5_path      : str    Destination file (see `soft_label_samples_path`).
    soft_labels  : array-like, shape (S, N, T) where S = n_chains * n_draws.
                   The `p_rfi` returned by `postprocessing_utils.classify`,
                   whose sample axis came from stacking ('chain', 'draw') in
                   that order -- so it reshapes to (n_chains, n_draws, N, T).
    n_chains     : int    Chains remaining after stuck-chain removal.
    n_draws      : int    Draws per chain.
    pointing     : int
    tv_subband   : str
    polarization : str
    data_dict    : dict   Padded per-night data; supplies night_ids, times,
                   lengths and mask so the file is self-describing.
    thin         : int    Keep every `thin`-th draw within each chain (1 = all).
    source_file  : str, optional  Chains file the labels were derived from.
    compression  : str or None    h5py compression filter for the big array.
    """
    S = soft_labels.shape[0]
    if n_chains * n_draws != S:
        raise ValueError(
            f"soft_labels has {S} samples, but n_chains*n_draws = "
            f"{n_chains}*{n_draws} = {n_chains * n_draws}. The sample axis must "
            "unstack exactly into (chain, draw)."
        )

    draw_idx     = np.arange(0, n_draws, thin)
    n_kept       = len(draw_idx)
    _, N, T      = soft_labels.shape
    nbytes       = n_chains * n_kept * N * T * 4

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    print(f"  Writing full soft labels: ({n_chains}, {n_kept}, {N}, {T}) "
          f"float32 ~ {nbytes / 1e9:.1f} GB uncompressed -> {h5_path}")

    with h5py.File(h5_path, 'w') as f:
        dset = f.create_dataset(
            'soft_labels',
            shape       = (n_chains, n_kept, N, T),
            dtype       = 'float32',
            chunks      = (1, min(n_kept, 32), N, T),
            compression = compression,
        )
        for c in range(n_chains):
            chain_block = np.asarray(soft_labels[c * n_draws:(c + 1) * n_draws],
                                     dtype='float32')
            dset[c] = chain_block[draw_idx]
            del chain_block

        _write_dataset(f, 'night_ids',
                       np.array(data_dict['night_ids'], dtype=h5py.special_dtype(vlen=str)))
        _write_dataset(f, 'time_stamps', np.asarray(data_dict['times']))
        _write_dataset(f, 'lengths',     np.asarray(data_dict['lengths']))
        _write_dataset(f, 'mask',        np.asarray(data_dict['mask']))

        f.attrs['pointing']     = pointing
        f.attrs['tv_subband']   = tv_subband
        f.attrs['polarization'] = polarization
        f.attrs['n_chains']     = n_chains
        f.attrs['n_draws']      = n_draws
        f.attrs['n_draws_kept'] = n_kept
        f.attrs['thin']         = thin
        f.attrs['source_file']  = source_file or ''

    print(f"  Written soft-label samples ({os.path.getsize(h5_path) / 1e9:.1f} GB on disk)")


def _write_scalar(group, name, value):
    """Write or overwrite a scalar dataset."""
    if name in group:
        del group[name]
    group.create_dataset(name, data=value)


def _write_dataset(group, name, array):
    """Write or overwrite an array dataset."""
    if name in group:
        del group[name]
    group.create_dataset(name, data=array)


GLOBAL_PARAMS   = ['w_clean', 'loc_rfi', 'scale_rfi']
GLOBAL_PARAM_STATS = ['median', 'hdi_94_lo', 'hdi_94_hi', 'hdi_50_lo', 'hdi_50_hi']


def read_season_results(h5_path):
    """
    Read the shared season_results.h5 file into one flat, tidy dataframe,
    with one row per (pol, tv_subband, pointing, night, timestep).

    Per-timestep columns include the classification labels (hard_label,
    soft_label, soft_label_std -- the posterior mean/std of the RFI
    responsibility p_rfi, i.e. the soft label's Bayesian error bar) as well
    as the reconstructed data itself -- the posterior median and std of the
    background-subtracted residual (residual_median, residual_std) -- so
    downstream analysis doesn't need to re-run MCMC +
    `postprocessing_utils.get_bg_residuals` just to see what the model saw.
    residual_median/residual_std/soft_label_std are NaN for files written
    before those fields existed.

    Per-night quantities (chisq_*, nll_*) and per-combo quantities
    (w_clean_*, loc_rfi_*, scale_rfi_*, tv_order) are broadcast across every
    timestep row they apply to, so the whole season can be filtered/grouped
    from this single object without any further merging. To recover
    night- or combo-level aggregates, `.drop_duplicates()` on the
    corresponding key columns (e.g. ['pol', 'tv_subband', 'pointing', 'night']),
    or use `.groupby(...)` for derived stats like empirical RFI fraction
    (`df.groupby([...])['hard_label'].mean()`).
    """
    records = []

    with locked_hdf5(h5_path, mode='r') as f:
        for pol in POLARIZATIONS:
            for tv in TV_SUBBANDS:
                for p in POINTINGS + [POOLED_POINTING]:
                    root = f"/{p}/{tv}/{pol}"
                    if root not in f:
                        if p != POOLED_POINTING:
                            print(f"No match found for {tv}, {p}, {pol}. Continuing...")
                        continue
                    grp = f[root]
                    ps  = grp['posterior_summary']

                    # ---- per-combo global posterior summaries (broadcast) --
                    global_vals = {}
                    for key in GLOBAL_PARAMS:
                        pgrp = ps[key]
                        for stat in GLOBAL_PARAM_STATS:
                            global_vals[f'{key}_{stat}'] = pgrp[stat][()]

                    # ---- per-night chi2 / nll, keyed by night id -----------
                    cg = ps['chisq_per_night']
                    chi2_by_night = {
                        _night_key(night): (cg['median'][n], cg['hdi_94_lo'][n], cg['hdi_94_hi'][n])
                        for n, night in enumerate(cg['night_ids'].asstr()[()])
                    }
                    ng = ps['rfi_nll_per_night']
                    nll_by_night = {
                        _night_key(night): (ng['median'][n], ng['hdi_94_lo'][n], ng['hdi_94_hi'][n])
                        for n, night in enumerate(ng['night_ids'].asstr()[()])
                    }

                    # ---- per-timestep labels --------------------------------
                    lg          = grp['labels']
                    hard_labels = np.asarray(lg['hard_labels'])  # (N_nights, T_max)
                    soft_labels = np.asarray(lg['soft_labels'])
                    times       = np.asarray(lg['time_stamps'])
                    lengths     = np.asarray(lg['lengths'])      # (N_nights,)
                    night_ids   = [_night_key(n) for n in lg['night_ids'].asstr()[()]]

                    # Files written before residuals/soft_labels_std were
                    # persisted won't have these datasets -- fall back to NaN.
                    if 'residuals_median' in lg:
                        residual_median = np.asarray(lg['residuals_median'])
                        residual_std    = np.asarray(lg['residuals_std'])
                    else:
                        residual_median = np.full(hard_labels.shape, np.nan)
                        residual_std    = np.full(hard_labels.shape, np.nan)

                    if 'soft_labels_std' in lg:
                        soft_labels_std = np.asarray(lg['soft_labels_std'])
                    else:
                        soft_labels_std = np.full(hard_labels.shape, np.nan)

                    for night_idx, (night, length) in enumerate(zip(night_ids, lengths)):
                        chi2_med, chi2_lo, chi2_hi = chi2_by_night.get(
                            night, (np.nan, np.nan, np.nan))
                        nll_med, nll_lo, nll_hi = nll_by_night.get(
                            night, (np.nan, np.nan, np.nan))

                        for t in range(length):
                            record = {
                                'pol'             : pol,
                                'tv_subband'      : tv,
                                'pointing'        : p,
                                'night'           : night,
                                'timestep'        : t,
                                'time_jd'         : times[night_idx, t],
                                'hard_label'      : hard_labels[night_idx, t],
                                'soft_label'      : soft_labels[night_idx, t],
                                'soft_label_std'  : soft_labels_std[night_idx, t],
                                'residual_median' : residual_median[night_idx, t],
                                'residual_std'    : residual_std[night_idx, t],
                                'chisq_median'    : chi2_med,
                                'chisq_hdi_94_lo' : chi2_lo,
                                'chisq_hdi_94_hi' : chi2_hi,
                                'nll_median'      : nll_med,
                                'nll_hdi_94_lo'   : nll_lo,
                                'nll_hdi_94_hi'   : nll_hi,
                            }
                            record.update(global_vals)
                            records.append(record)

    if not records:
        raise FileNotFoundError(f"No results found in {h5_path}")

    df = pd.DataFrame(records)
    df['tv_order'] = df['tv_subband'].map({tv: i for i, tv in enumerate(TV_SUBBANDS)})

    n_combos = df[['pol', 'tv_subband', 'pointing']].drop_duplicates().shape[0]
    print(f"Loaded {n_combos} (pol, tv_subband, pointing) combinations "
          f"({len(df)} timestep rows) from {h5_path}")

    return df


def read_rejected_obsids(h5_path):
    """
    Read every (pointing, tv_subband) combination's rejected-night obsids
    from the shared season_results.h5 file.

    These are nights excluded from the mixture-model fit (via
    `mixture_model.reject_nights`) for exceeding the SSINS flagged-fraction
    tolerance, and therefore never appear in the per-timestep records from
    `read_season_results` -- there's no posterior classification to look up
    for them. Downstream flag-propagation should flag them outright instead
    (see `propagate.py`).

    Returns
    -------
    rejected : dict
        Maps (pointing, tv_subband) -> list of obsid strings. Combos with no
        rejected nights, or not yet written to `h5_path`, are omitted.
    """
    rejected = {}

    with locked_hdf5(h5_path, mode='r') as f:
        for tv in TV_SUBBANDS:
            for p in POINTINGS + [POOLED_POINTING]:
                root = f"/{p}/{tv}"
                if root not in f or 'rejected_obsids' not in f[root]:
                    continue
                obsids = list(f[root]['rejected_obsids'].asstr()[()])
                if obsids:
                    rejected[(p, tv)] = obsids

    n_obsids = sum(len(v) for v in rejected.values())
    print(f"Loaded {n_obsids} rejected obsid(s) across {len(rejected)} "
          f"(pointing, tv_subband) combination(s) from {h5_path}")

    return rejected
