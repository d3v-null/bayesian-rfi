import arviz as az
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import yaml

import mixture_model


def load_input_data(L, abs_dir, tv_subband, pointing, polarization, keep_night_ids=None, scale=1.0):
    """
    Load and package one (tv_subband, pointing, polarization) combination's
    nightly data for postprocessing, mirroring the inputs the model was fit on.

    Rebuilds the candidate night list `fit.py` used (via
    `mixture_model.build_dataset`, which applies the same flagged-fraction
    rejection), loads the raw per-night .npz files, computes the analytic
    thermal noise `sigma_clean`, and pads everything into the model's
    `(N_nights, T_max)` pytree via `mixture_model.prepare_multiple_nights`.

    Parameters
    ----------
    L : int
        Number of Legendre coefficients (must match the value used to fit
        the model).
    abs_dir : str
        Absolute path to the project root, with a trailing slash (used for
        direct string concatenation, not `os.path.join`).
    tv_subband : str
        TV sub-band identifier (e.g. 'TV7').
    pointing : int
        Pointing index.
    polarization : str
        Polarization to load ('xx' or 'yy').
    keep_night_ids : list of str, optional
        The exact night_ids the model was fit on, IN FIT ORDER (i.e. read
        line-by-line from the `..._nights.txt` sidecar file
        `fit.py` saves next to each chains file -- don't
        re-sort it). If given, restricts to exactly these nights, in this
        exact order, via `mixture_model.select_nights_by_id` (order matters:
        it's what lines up each night with its `beta` in the posterior),
        raising if any can't be found among this combo's current candidate
        nights. If None, uses the full (rejection-filtered) season -- only
        correct for a run that wasn't
        split (`split_method=None` at fit time).
    scale : float, optional
        Factor the data were rescaled by at fit time (`fit.py --rescale-noise`,
        saved in the `_scale.txt` sidecar). Default 1, no rescaling.

    Returns
    -------
    sigma_clean : jnp.ndarray, shape (N,)
        Per-night analytic thermal noise standard deviation.
    data_dict : dict
        Padded per-night data pytree from `mixture_model.prepare_multiple_nights`
        (keys: y_pad, A_pad, f_pad, mask, lengths, y_loc, y_scale, night_ids,
        obsids, times).
    rejected_obsids : dict
        Night ID -> list of full MWA obsid strings, for nights excluded from
        this (pointing, tv_subband) combo for excessive SSINS flagging. See
        `mixture_model.reject_nights`.
    """
    all_files, rejected_nights, night_ids, rejected_obsids = mixture_model.build_dataset(
        pointing      = pointing,
        tv_subband    = tv_subband,
        abs_directory = abs_dir,
    )

    print(f"Rejected {len(rejected_nights)} nights: {rejected_nights}")

    if keep_night_ids is not None:
        all_files, night_ids = mixture_model.select_nights_by_id(all_files, night_ids, keep_night_ids)
        print(f"Restricted to {len(all_files)} saved night(s) from the fit's night list")

    print(f"Remaining: {len(all_files)} nights\n")

    all_data    = [np.load(f) for f in all_files]
    raw_samples = [d[polarization] for d in all_data]
    ssin_flags = [d["ssins"] for d in all_data]
    times       = [d['time_jd'] for d in all_data]
    obsids      = [d['obsids'] for d in all_data]

    with open(abs_dir+'data/N_channels.yaml', "r") as f:
        tv_subband_dict = yaml.safe_load(f)

    nbls   = [int(d['nbls'])  for d in all_data] if all('nbls'  in d for d in all_data) else None
    nchans = [int(d['nchan']) for d in all_data] if all('nchan' in d for d in all_data) else None
    sigma_clean = mixture_model.calculate_sigma_clean(raw_samples, tv_subband_dict=tv_subband_dict, tv_subband=tv_subband,
                                                      nbls=nbls, nchans=nchans)
    if scale != 1.0:
        raw_samples = [np.asarray(s) * scale for s in raw_samples]
        sigma_clean = sigma_clean * scale
    data_dict = mixture_model.prepare_multiple_nights(raw_samples,
                                                    L=L,
                                                    flags=ssin_flags,
                                                    night_ids=night_ids,
                                                    obsids=obsids,
                                                    times=times
                                                    )
    return sigma_clean, data_dict, rejected_obsids


def extract_samples(idata):
    """
    Flatten posterior samples from an arviz InferenceData into per-parameter
    arrays with chain and draw collapsed into a single sample dimension.

    Parameters
    ----------
    idata : arviz.InferenceData
        Fitted model posterior, with `chain` and `draw` dimensions.

    Returns
    -------
    dict with keys:
        beta      : jnp.ndarray, shape (S, N, L)  per-night Legendre coefficients
        w_clean   : jnp.ndarray, shape (S,)        global clean mixture weight
        loc_rfi   : jnp.ndarray, shape (S,)        global RFI component location
        scale_rfi : jnp.ndarray, shape (S,)        global RFI component scale
    where S = n_chains * n_draws, N = n_nights, L = number of Legendre coefficients.
    """
    beta      = idata.posterior['beta'].stack(sample=('chain','draw')).transpose('sample', ...).values
    w_clean   = idata.posterior['w_clean'].stack(sample=('chain','draw')).transpose('sample',...).values
    loc_rfi   = idata.posterior['loc_rfi'].stack(sample=('chain','draw')).transpose('sample', ...).values
    scale_rfi = idata.posterior['scale_rfi'].stack(sample=('chain','draw')).transpose('sample', ...).values

    return {
        "beta": beta,
        "w_clean": w_clean,
        "loc_rfi": loc_rfi,
        "scale_rfi": scale_rfi
    }


def classify(z, samples, sigma_clean):
    """
    Classify residuals as clean (0) or RFI (1) using posterior-averaged mixture parameters.

    Parameters
    ----------
    z : array-like, shape (S, N, T)
        Per-draw residuals after background subtraction.
    samples : dict
        Posterior samples from MCMC
    sigma_clean : array-like
        Calculated noise standard deviation, shape (N,)

    Returns
    -------
    p_rfi : jnp.ndarray, shape (S, N, T)
        RFI responsibility in [0, 1].
    labels : jnp.ndarray, shape (S, N, T), dtype int32
        Hard labels: 1 = RFI, 0 = clean (threshold at p_rfi > 0.5).
    """
    z           = jnp.asarray(z)
    sigma_clean = jnp.asarray(sigma_clean)
    w_clean     = jnp.asarray(samples['w_clean'])
    loc_rfi     = jnp.asarray(samples['loc_rfi'])
    scale_rfi   = jnp.asarray(samples['scale_rfi'])

    # Expand all parameters for broadcasting against (S, N, T)
    w_clean     = w_clean[:, None, None]                    # (S, 1, 1)
    w_rfi       = 1.0 - w_clean                             # (S, 1, 1)
    loc_rfi     = loc_rfi[:, None, None]                    # (S, 1, 1)
    scale_rfi   = scale_rfi[:, None, None]                  # (S, 1, 1)
    sigma_clean = sigma_clean[None, :, None]                # (1, N, 1)

    logp_clean = jsp.stats.norm.logpdf(z, loc=0.0, scale=sigma_clean)
    logp_rfi   = mixture_model.SkewCauchy(loc_rfi, scale_rfi, 0.99).log_prob(z)

    logw_clean = jnp.log(w_clean) + logp_clean
    logw_rfi   = jnp.log(w_rfi)   + logp_rfi

    # posterior responsibility
    p_rfi = jnp.exp(logw_rfi - jnp.logaddexp(logw_clean, logw_rfi))

    # hard labels
    labels = jnp.where(jnp.isnan(p_rfi), jnp.nan, (p_rfi > 0.5).astype(jnp.float32))

    return p_rfi, labels


def get_bg_residuals(data_dict, samples_dict):
    """
    Recompute the per-draw Legendre background and residuals from posterior
    beta samples. Padded (invalid) entries are set to NaN rather than 0 so
    they can't silently bias downstream aggregates.

    Parameters
    ----------
    data_dict : dict
        Padded per-night data, as returned by `mixture_model.prepare_multiple_nights`
        (needs A_pad, shape (N, T, L); y_pad, shape (N, T); mask, shape (N, T)).
    samples_dict : dict
        Posterior samples, as returned by `extract_samples` (needs beta,
        shape (S, N, L)).

    Returns
    -------
    mu_bg : jnp.ndarray, shape (S, N, T)
        Per-draw Legendre background prediction, NaN outside `mask`.
    z : jnp.ndarray, shape (S, N, T)
        Per-draw residual (y_pad - mu_bg), NaN outside `mask`.
    """

    A = data_dict['A_pad']
    mu_bg = jnp.einsum("snl,ntl->snt", samples_dict['beta'], A) 

    z = data_dict['y_pad'][None, :, :] - mu_bg

    mask = data_dict['mask']
    mu_bg = jnp.where(mask[None, :, :], mu_bg, jnp.nan)
    
    z = jnp.where(mask[None, :, :], z, jnp.nan)

    return mu_bg, z


def get_residual_summary(z):
    """
    Collapse per-draw residuals to a posterior median and std at every
    (night, timestep) -- the reconstructed data itself, as opposed to labels
    or metrics derived from it, for persisting to the season-results HDF5.

    Parameters
    ----------
    z : array-like, shape (S, N, T)
        Per-draw residuals after background subtraction (from
        `get_bg_residuals`). NaN outside the per-night mask.

    Returns
    -------
    residual_median : np.ndarray, shape (N, T)
        Posterior median residual per night/timestep.
    residual_std : np.ndarray, shape (N, T)
        Posterior std of the residual per night/timestep.
    """
    z = np.asarray(z)
    residual_median = np.nanmedian(z, axis=0)
    residual_std    = np.nanstd(z, axis=0)
    return residual_median, residual_std


def get_chi2(z, sigma_clean, clean_mask):
    """
    Compute the posterior median and 94% HDI of reduced chi-squared per
    night, over residuals classified as "clean" — a goodness-of-fit check
    for the clean (Normal) mixture component.

    Parameters
    ----------
    z : array-like, shape (S, N, T)
        Per-draw residuals after background subtraction (from `get_bg_residuals`).
    sigma_clean : array-like, shape (N,)
        Per-night analytic thermal noise standard deviation.
    clean_mask : array-like, broadcastable to (S, N, T), dtype bool
        True where a residual is labeled clean (p_rfi <= 0.5).

    Returns
    -------
    chi2_median : np.ndarray, shape (N,)
        Posterior median reduced chi-squared per night.
    chi2_lo, chi2_hi : np.ndarray, shape (N,)
        Lower/upper bounds of the 94% HDI per night.
    """
    sigma_c = sigma_clean[None, :, None]

    # Chi-squared contribution per sample
    chi2_terms = jnp.where(clean_mask, (z / sigma_c)**2, np.nan)   # (S, N, T)

    n_clean    = clean_mask.sum(axis=-1)                         # (S, N)
    chi2_reduced = jnp.nansum(chi2_terms, axis=-1) / n_clean     # (S, N)

    chi2_median = np.array(np.median(chi2_reduced, axis=0))  # (N,)
    hdi = np.array([az.hdi(np.array(chi2_reduced[:, n]), prob=0.94) for n in range(chi2_reduced.shape[1])])  # (N, 2)
    chi2_lo = hdi[:, 0]
    chi2_hi = hdi[:, 1]

    return chi2_median, chi2_lo, chi2_hi


def get_neg_logpdf_rfi(z, samples_dict, rfi_mask):
    """
    Compute the posterior median and 94% HDI of the per-night mean negative
    log-pdf of the RFI (SkewCauchy) component, over residuals classified as
    "RFI" — a goodness-of-fit check for the RFI mixture component.

    Parameters
    ----------
    z : array-like, shape (S, N, T)
        Per-draw residuals after background subtraction (from `get_bg_residuals`).
    samples_dict : dict
        Posterior samples, as returned by `extract_samples` (needs loc_rfi,
        scale_rfi, each shape (S,)).
    rfi_mask : array-like, broadcastable to (S, N, T), dtype bool
        True where a residual is labeled RFI (p_rfi > 0.5).

    Returns
    -------
    nll_median : np.ndarray, shape (N,)
        Posterior median mean negative log-pdf per night.
    nll_lo, nll_hi : np.ndarray, shape (N,)
        Lower/upper bounds of the 94% HDI per night.
    """

    # Posterior parameters expanded for broadcasting
    loc_rfi_s   = jnp.asarray(samples_dict['loc_rfi'])[:, None, None]    # (S, 1, 1)
    scale_rfi_s = jnp.asarray(samples_dict['scale_rfi'])[:, None, None]  # (S, 1, 1)

    # Negative log-pdf of SkewCauchy at each RFI residual
    nll_terms = -mixture_model.SkewCauchy(loc_rfi_s, scale_rfi_s, 0.99).log_prob(z)  # (S, N, T)

    # Sum over RFI samples per draw per night, normalise by count
    n_rfi        = rfi_mask.sum(axis=-1)                             # (S, N)
    nll_per_draw = jnp.where(rfi_mask, nll_terms, 0.0).sum(axis=-1) / n_rfi  # (S, N)

    # Posterior median + HDI per night
    nll_median = np.median(np.array(nll_per_draw), axis=0)           # (N,)
    hdi_nll    = np.array([az.hdi(np.array(nll_per_draw[:, n]), prob=0.94)
                        for n in range(nll_per_draw.shape[1])])   # (N, 2)
    nll_lo = hdi_nll[:, 0]
    nll_hi = hdi_nll[:, 1]

    return nll_median, nll_lo, nll_hi


def posterior_summary(samples, key, hdi_prob_outer=0.94, hdi_prob_inner=0.50):
    """
    Compute posterior median and HDI intervals for a given parameter.

    Parameters
    ----------
    samples : dict
        Posterior samples from MCMC.
    key : str
        Parameter name to summarise.
    hdi_prob_outer : float, optional
        Outer HDI probability. Default 0.94.
    hdi_prob_inner : float, optional
        Inner HDI probability. Default 0.50.

    Returns
    -------
    median : float
    hdi_outer : np.ndarray, shape (2,)
    hdi_inner : np.ndarray, shape (2,)
    """
    s = np.array(samples[key]).flatten()
    median    = float(np.median(s))
    hdi_outer = az.hdi(s, prob=hdi_prob_outer)
    hdi_inner = az.hdi(s, prob=hdi_prob_inner)
    return median, hdi_outer, hdi_inner
