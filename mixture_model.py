import glob
import os
import re

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpy.polynomial.legendre import legvander

# ----------------------------
# Helpers
# ----------------------------


def reject_nights(npz_files, tol=0.33):
    """
    Remove nights where the fraction of flagged samples exceeds a tolerance.

    Parameters
    ----------
    npz_files : list of str
        Paths to .npz files, one per night.
    tol : float, optional
        Maximum allowed flagged fraction. Nights where nansum(ssins)/ssins.size > tol
        are rejected. Default 0.33.

    Returns
    -------
    npz_files : list of str
        Filtered file paths with rejected nights removed.
    reject : list of str
        Night IDs of rejected nights.
    night_ids: list of str
        Night IDs of nights kept.
    reject_obsids : dict
        Night ID -> list of full MWA obsid strings, for rejected nights only.
        A night's npz file is keyed only by its first obsid's truncated ID,
        so this is what lets downstream flag-propagation identify every
        UVFlag file (one per obsid) that belongs to a rejected night.
    """
    reject = []
    reject_obsids = {}
    for f in npz_files:
        d = np.load(f)
        flags = d["ssins"]
        if np.nansum(flags) / flags.size > tol:
            night_id = re.search(r'\d{6}', f).group()
            reject.append(night_id)
            reject_obsids[night_id] = [str(o) for o in d["obsids"]]

    npz_files = [f for f in npz_files if not any(obs_id in f for obs_id in reject)]
    night_ids = [re.search(r"\d{6}", f).group() for f in npz_files]

    return npz_files, reject, night_ids, reject_obsids


def chrono_split_nights(npz_files, night_ids, half):
    """
    Restrict to the first or second half of a season of nights, chronologically.

    Parameters
    ----------
    npz_files : list of str
        Paths to .npz files, one per night (aligned with night_ids).
    night_ids : list of str
        Night IDs aligned with npz_files. These are truncated (6-digit) MWA
        obsids, which increase monotonically with time.
    half : int
        0 or 1. Nights are sorted chronologically by night_id and split at
        the midpoint: half 0 is the first half of the season, half 1 is the
        second half.

    Returns
    -------
    npz_files : list of str
        Filtered file paths, restricted to the requested half.
    night_ids : list of str
        Filtered night IDs, restricted to the requested half.
    """
    n_nights = len(npz_files)
    order = np.argsort([int(n) for n in night_ids])
    midpoint = n_nights // 2
    half_idx = order[:midpoint] if half == 0 else order[midpoint:]

    npz_files = [npz_files[i] for i in half_idx]
    night_ids = [night_ids[i] for i in half_idx]

    return npz_files, night_ids


def random_split_nights(npz_files, night_ids, half, random_seed):
    """
    Restrict to a uniformly random half of a season of nights.

    Parameters
    ----------
    npz_files : list of str
        Paths to .npz files, one per night (aligned with night_ids).
    night_ids : list of str
        Night IDs aligned with npz_files.
    half : int
        0 or 1. Selects which half of the shuffled nights to keep.
    random_seed : int
        Seed for the shuffle, so the split is reproducible *within this
        process*. Note this is not a substitute for saving the resulting
        night_ids out (see `fit.py`) -- shuffles aren't
        guaranteed reproducible across numpy versions/platforms, so
        postprocessing must replay the exact saved list (`select_nights_by_id`)
        rather than re-deriving the split from this function.

    Returns
    -------
    npz_files : list of str
        Filtered file paths, restricted to the requested half.
    night_ids : list of str
        Filtered night IDs, restricted to the requested half.
    """
    rng = np.random.RandomState(random_seed)
    n_nights = len(npz_files)
    order = np.arange(n_nights)
    rng.shuffle(order)
    midpoint = n_nights // 2
    half_idx = order[:midpoint] if half == 0 else order[midpoint:]

    npz_files = [npz_files[i] for i in half_idx]
    night_ids = [night_ids[i] for i in half_idx]

    return npz_files, night_ids


def select_nights_by_id(npz_files, night_ids, keep_night_ids):
    """
    Restrict npz_files/night_ids to an explicit, previously-chosen list of
    night IDs, in that exact order.

    This is how postprocessing replays the *exact* nights (and their exact
    order) a model was fit on, regardless of how they were originally
    selected (full season, chronological split, random split, ...) --
    rather than re-deriving a split, which is fragile (a random shuffle
    isn't guaranteed reproducible across numpy versions/platforms, and even
    a deterministic split can silently diverge if the underlying candidate
    nights change between fit time and postprocess time, e.g. new raw data
    arriving or a different rejection tolerance). See
    `fit.py`, which saves the final night_ids list next to
    each chains file, in fit order.

    Parameters
    ----------
    npz_files : list of str
        Paths to .npz files, one per night (aligned with night_ids).
    night_ids : list of str
        Night IDs aligned with npz_files.
    keep_night_ids : sequence of str
        Night IDs to keep, IN THE EXACT ORDER they were used at fit time
        (i.e. as read line-by-line from the `..._nights.txt` sidecar). Order
        matters and is preserved in the output: `beta`'s per-night posterior
        index is tied to whatever order the model was fit with, so if a
        night's data ends up at a different position than it was fit at
        (e.g. re-sorted chronologically after a random split), the
        reconstructed background/residuals get silently computed against
        the wrong night's posterior.

    Returns
    -------
    npz_files : list of str
        File paths reordered to match keep_night_ids exactly.
    night_ids : list of str
        Equal to keep_night_ids (as a list), for convenience.

    Raises
    ------
    ValueError
        If any of keep_night_ids isn't found among npz_files/night_ids --
        the saved list can no longer be exactly reconstructed (e.g. the
        night was rejected under a different tolerance, or its data is
        missing), so silently proceeding with a partial/mismatched set
        would risk quietly corrupting the reconstructed data instead of
        failing loudly.
    """
    lookup = dict(zip(night_ids, npz_files))
    keep_night_ids = list(keep_night_ids)

    missing = [n for n in keep_night_ids if n not in lookup]
    if missing:
        raise ValueError(
            f"Night ID(s) {sorted(missing)} from the saved night list were not found "
            "among this combo's candidate nights -- the data or rejection tolerance may "
            "have changed since the model was fit. Cannot exactly reconstruct the fit's input."
        )

    new_files = [lookup[n] for n in keep_night_ids]
    return new_files, keep_night_ids


def build_dataset(pointing, tv_subband, abs_directory, tol=0.33):
    """
    Build a list of .npz data files for a given pointing and TV subband,
    dropping nights with excessive flagging.

    Parameters
    ----------
    pointing : int
        Pointing index (one of 0, 1, 2, 3, 4).
    tv_subband : str
        TV subband string (one of 'subTV', 'TV6', 'TV7', 'TV8', 'TV9').
    abs_directory : str
        Absolute path to the root data directory.
    tol : float, optional
        Flagged fraction tolerance passed to reject_nights. Default 0.33.

    Returns
    -------
    all_files : list of str
        Paths to retained .npz files.
    rejected : list of str
        Night IDs of nights rejected due to excessive flagging.
    night_ids: list of str
        Night IDs of nights kept.
    reject_obsids : dict
        Night ID -> list of full MWA obsid strings, for rejected nights only.
        See `reject_nights`.
    """
    all_files = sorted(glob.glob(
        os.path.join(abs_directory, f"data/tv_split/{tv_subband}/processed_*_p{pointing}.npz")
    ))

    if not all_files:
        print(f"Warning: no files found for subband {tv_subband}, pointing {pointing}")

    return reject_nights(all_files, tol=tol)


def map_to_unit_interval(u):
    """
    Map a 1D coordinate array to [-1, 1].
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    umin = float(np.min(u))
    umax = float(np.max(u))

    return 2.0 * (u - umin) / (umax - umin) - 1.0


def legendre_design_from_time(u, L):
    """
    Build a Legendre design matrix from u with L coefficients

    Returns
    -------
    A : jnp.ndarray, shape (T, L)
        Design matrix with columns P_0 ... P_{L-1}.
    x_scaled : jnp.ndarray, shape (T,)
        The mapped coordinates in [-1, 1].
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)

    x_scaled = map_to_unit_interval(u)
    A = legvander(x_scaled, L - 1)   # shape (T, L)

    return jnp.asarray(A), jnp.asarray(x_scaled)


def prepare_single_night(sample, L, flags, times):
    """
    Prepare one night's data:
      - flatten to 1D
      - remove non-finite values
      - preserve ORIGINAL sample indices for gappy time coordinates
      - build Legendre matrix on those original indices
    """
    raw = np.asarray(sample, dtype=np.float64).reshape(-1)

    valid_mask = np.isfinite(raw)
    y = raw[valid_mask]
    t = times[valid_mask]
    f = flags[valid_mask]
    u = np.flatnonzero(valid_mask).astype(np.float64)  # preserves gaps
    A, x_scaled = legendre_design_from_time(u, L)

    return y, t, A, f, x_scaled, valid_mask, u



def prepare_multiple_nights(samples, L, flags, night_ids, obsids, times):
    """
    Prepare a list of ragged nightly arrays into padded JAX pytrees
    suitable for the hierarchical model.

    Parameters
    ----------
    samples : list of array-like
        Each element is one night's raw 1-D data (may contain NaNs/infs,
        may differ in length).
    L : int
        Number of Legendre coefficients.
    flags : list of array-like
        SSINS flags associated with samples
    night_ids: list of str
        Night IDs of nights in data (aligned with samples)

    Returns
    -------
    dict with keys:
        y_pad   : jnp.ndarray  (N_nights, T_max)   padded observations
        A_pad   : jnp.ndarray  (N_nights, T_max, L) padded design matrices
        mask    : jnp.ndarray  (N_nights, T_max)    True where data are valid
        lengths : list[int]    actual length per night (for reference)
        y_loc   : float        global location prior (grand mean)
        y_scale : float        global scale prior  (grand std)
    """
    nights = [prepare_single_night(s, L, f, t) for s,f,t in zip(samples, flags, times)]

    lengths = [len(n[0]) for n in nights]
    T_max = max(lengths)
    N = len(nights)

    y_pad     = np.zeros((N, T_max), dtype=np.float64)
    times_pad = np.zeros((N, T_max), dtype=np.float64)
    A_pad     = np.zeros((N, T_max, L), dtype=np.float64)
    f_pad     = np.zeros((N, T_max), dtype=bool)
    mask      = np.zeros((N, T_max), dtype=bool)
    y_loc     = np.zeros(N, dtype=np.float64)   # per-night
    y_scale   = np.zeros(N, dtype=np.float64)  # per-night

    for i, (y, t, A, f, *_) in enumerate(nights):
        T_i = len(y)
        y_pad[i, :T_i]    = y
        times_pad[i,:T_i] = t
        A_pad[i, :T_i, :] = A
        f_pad[i, :T_i]    = f
        mask[i,  :T_i]    = True
        y_loc[i]          = float(np.median(y))
        y_scale[i]        = float(np.std(y))

    return {
        "y_pad": jnp.asarray(y_pad),
        "A_pad": jnp.asarray(A_pad),
        "f_pad": jnp.asarray(f_pad),
        "mask": jnp.asarray(mask),
        "lengths": jnp.asarray(lengths),
        "y_loc": jnp.asarray(y_loc),
        "y_scale": jnp.asarray(y_scale),
        "night_ids": np.asarray(night_ids),
        "obsids": obsids,
        "times": np.asarray(times_pad, dtype=np.float64),
    }


def calculate_sigma_clean(samples, tv_subband_dict, tv_subband, nbls=None, nchans=None):
    """
    Calculate the expected thermal noise standard deviation for each night.

    Parameters
    ----------
    samples : list of array-like
        Raw nightly data arrays.
    tv_subband_dict : dict
        Sub-band -> number of fine channels averaged (fallback when `nchans` is None).
    nbls : sequence of int, optional
        Per-night number of baselines averaged into the INS (from the npz
        `nbls` key). Falls back to the Phase I value 128*127/2 if None.
    nchans : sequence of int, optional
        Per-night number of channels averaged (npz `nchan` key).

    Returns
    -------
    sigma_clean : jnp.ndarray, shape (N,)
        Per-night thermal noise standard deviation.
    """
    C = 4/np.pi - 1
    N = len(samples)
    if nbls is None:
        nbls = [128*127/2] * N  # Nbl = 8128, Phase I with no flagged tiles
    if nchans is None:
        nchans = [tv_subband_dict[tv_subband]] * N  # Channels averaged together

    sigma_clean = []
    for sample, Nbl, Nchannels in zip(samples, nbls, nchans):
        valid_mask = np.isfinite(sample)
        valid_sample = sample[valid_mask]  # only consider valid samples
        mu_hat = np.median(valid_sample)

        sigma_clean.append(np.sqrt((C*mu_hat**2)/(Nbl * Nchannels)))

    return jnp.asarray(sigma_clean)


# ----------------------------------
# Custom Distributions
# ----------------------------------


class SkewCauchy(dist.Distribution):
    """
    Skew-Cauchy distribution matching scipy.stats.skewcauchy

    f(x, a) = 1 / (pi * (x^2 / (a*sign(x) + 1)^2 + 1))

    for skewness parameter -1 < a < 1. When a=0 reduces to standard Cauchy.
    """
    arg_constraints = {  # noqa: RUF012
        'loc'  : dist.constraints.real,
        'scale': dist.constraints.positive,
        'a'    : dist.constraints.open_interval(-1.0, 1.0),  # strictly (-1, 1)
    }
    support = dist.constraints.real

    def __init__(self, loc, scale, a, validate_args=None):
        self.loc, self.scale, self.a = loc, scale, a
        super().__init__(batch_shape=jnp.shape(loc), validate_args=validate_args)

    def log_prob(self, z):
        x = (z - self.loc) / self.scale
        denom_scale = self.a * jnp.sign(x) + 1.0
        return -jnp.log(jnp.pi) - jnp.log(x**2 / denom_scale**2 + 1.0) - jnp.log(self.scale)


# ----------------------------
# Model
# ----------------------------


def model(A_pad, y_pad, mask, y_loc, y_scale, sigma_clean, rfi_scale_cutoff):
    """
    Hierarchical 2-component mixture model over multiple nights.

    Parameters
    ----------
    A_pad          : jnp.ndarray, shape (N, T_max, L)
        Padded Legendre design matrices.
    y_pad          : jnp.ndarray, shape (N, T_max)
        Padded observations.
    mask           : jnp.ndarray, shape (N, T_max), dtype bool
        True for valid observations; False for padding.
    y_loc          : jnp.ndarray, shape (N,)
        Per-night median, used as prior location for the zeroth Legendre coefficient.
    y_scale        : jnp.ndarray, shape (N,)
        Per-night std, used as prior scale for the zeroth Legendre coefficient.
    sigma_clean    : jnp.ndarray, shape (N,)
        Per-night thermal noise standard deviation, empirically derived from data.
    rfi_scale_cutoff : float
        Hard lower bound on the SkewCauchy scale parameter. scale_rfi is
        sampled from LogNormal(0, 0.5) shifted by this value.

    Hierarchy
    ---------
    Global (shared across all nights):
        w_rfi      ~ Beta(2, 10)
        w_clean     = 1 - w_rfi
        loc_rfi   ~ HalfNormal(0.25)
        scale_rfi ~ LogNormal(0, 0.5) + rfi_scale_cutoff
        mu_beta   ~ Normal(0, 1)^(L-1)    hyperprior loc per Legendre mode L=1..L-1
        sigma_beta ~ HalfNormal(1)^(L-1)  hyperprior scale per Legendre mode L=1..L-1

    Per-night (plate "nights", dim=-1):
        beta_n ~ Normal(
                     [y_loc_n,        mu_beta   ],
                     [5 * y_scale_n,  sigma_beta]
                 )^L

    Per-observation (plates "nights" x "timesteps", masked):
        z_nt        = y_nt - A_pad[n] @ beta_n
        log p(z_nt) = log( w_clean * Normal(0, sigma_clean_n).pdf(z_nt)
                         + w_rfi   * SkewCauchy(loc_rfi, scale_rfi, a=0.99).pdf(z_nt) )

    Padded entries (mask=False) contribute 0 to the log-likelihood.
    """
    A_pad   = jnp.asarray(A_pad)
    y_pad   = jnp.asarray(y_pad)
    mask    = jnp.asarray(mask)
    y_loc   = jnp.asarray(y_loc)
    y_scale = jnp.asarray(y_scale)

    N, T_max, L = A_pad.shape
    degree      = L - 1

    # ---- Global parameters ------------------------------------------------
    w_rfi = numpyro.sample("w_rfi", dist.Beta(2, 10))
    w_clean = numpyro.deterministic("w_clean", 1.0 - w_rfi)

    rfi_scale_cutoff = numpyro.deterministic("scale_rfi_cutoff", rfi_scale_cutoff)

    loc_rfi   = numpyro.sample("loc_rfi",   dist.HalfNormal(0.25))
    scale_rfi = numpyro.sample(
        "scale_rfi",
        dist.TransformedDistribution(
            dist.LogNormal(0, 0.5),
            dist.transforms.AffineTransform(loc=rfi_scale_cutoff, scale=1.0)
        )
    )

    # One mu and sigma per mode, shared across all nights
    mu_beta    = numpyro.sample(
        "mu_beta",
        dist.Normal(jnp.zeros(degree), jnp.ones(degree)).to_event(1),
    )  # (degree,)

    sigma_beta = numpyro.sample(
        "sigma_beta",
        dist.HalfNormal(jnp.ones(degree)).to_event(1),
    )  # (degree,)

    y_loc   = numpyro.deterministic("mu_beta0", y_loc)
    y_scale = numpyro.deterministic("sigma_beta0", y_scale)

    # ---- Per-night parameters + likelihood --------------------------------
    with numpyro.plate("nights", N, dim=-1):

        beta = numpyro.sample(
            "beta",
            dist.Normal(
                jnp.concatenate([y_loc[:, None],
                                 jnp.broadcast_to(mu_beta, (N, degree))],    axis=-1),  # (N, L)
                jnp.concatenate([5.0 * y_scale[:, None],
                                 jnp.broadcast_to(sigma_beta, (N, degree))], axis=-1),  # (N, L)
            ).to_event(1),
        )  # (..., N, L)

    mu_bg = jnp.einsum("...nl,ntl->...nt", beta, A_pad)   # (..., N, T_max)
    z     = y_pad - mu_bg                                 # (..., N, T_max)

    with numpyro.plate("nights", N, dim=-2):  # noqa: SIM117
        with numpyro.plate("timesteps", T_max, dim=-1):
            log_prob_clean = dist.Normal(0.0, sigma_clean[:, None]).log_prob(z)  # (..., N, T_max)
            log_prob_rfi   = SkewCauchy(loc=loc_rfi, scale=scale_rfi, a=0.99).log_prob(z)
            log_prob = jnp.logaddexp(
                jnp.log(w_clean) + log_prob_clean,
                jnp.log(w_rfi)   + log_prob_rfi,
            )
            masked_log_prob = jnp.where(mask, log_prob, 0.0)  # Remove padded entries
            numpyro.factor("log_likelihood", masked_log_prob)
