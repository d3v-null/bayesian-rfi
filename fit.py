import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

# Calculate time elapsed
T_START = time.perf_counter()

import arviz as az
import cli
import numpy as np
import numpyro
import yaml
from numpyro.handlers import reparam
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.reparam import LocScaleReparam

import mixture_model

# One host device per chain, capped by the cores actually available.
N_CHAINS = min(48, os.cpu_count() or 1)
numpyro.set_host_device_count(N_CHAINS)

from jax import random as jrandom

ABS_DIR       = os.environ.get("BAYESRFI_ABS_DIR", "/oscar/data/jpober/jmduchar/Research/BayesRFI/")
TV_SUBBANDS   = ['subTV', 'TV6', 'TV7', 'TV8', 'TV9']
POINTINGS     = [0, 1, 2, 3, 4]
POLARIZATIONS = ['xx', 'yy']
L             = 24
RANDOM_SEED   = 42

JOB_MAP = {
    idx: (L, tv, p, pol)
    for idx, (tv, p, pol) in enumerate(
        (tv, p, pol)
        for tv   in TV_SUBBANDS
        for p    in POINTINGS
        for pol  in POLARIZATIONS
    )
}

# `--pool-pointings`: one task per (sub-band, pol), every pointing in one fit
POOLED_JOB_MAP = {
    idx: (L, tv, mixture_model.POOLED_POINTING, pol)
    for idx, (tv, pol) in enumerate(
        (tv, pol)
        for tv   in TV_SUBBANDS
        for pol  in POLARIZATIONS
    )
}

# NUTS settings of the paper (Section IV); overridable for quick test runs
NUM_WARMUP  = 6000
NUM_SAMPLES = 3000

def run_mixture_model(pointing, L, pol, tv_subband, random_seed, run_dir,
                    split_method=None, half=None,
                    num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, num_chains=N_CHAINS,
                    rescale_noise=None, non_centred=False):
    """
    non_centred : bool, optional
        Sample the per-night Legendre coefficients non-centred
        (`beta = loc + scale * beta_decentered`). The same model; it removes the
        mu_beta/sigma_beta funnel that makes NUTS diverge when there are only a
        few nights (e.g. the blocks of one night). `beta` is still recorded.
    rescale_noise : float, optional
        If given, rescale the data so the median noise is this value (see
        `mixture_model.rescale_to_noise`); the factor is saved in a `_scale.txt`
        sidecar for postprocessing.
    pointing : int or str
        Pointing index, or `mixture_model.POOLED_POINTING` to fit every
        pointing's (night, pointing) blocks together.
    split_method : {None, 'chrono', 'random'}, optional
        If given (together with `half`), restricts the fit to half the
        season: 'chrono' splits nights at the midpoint by time, 'random'
        shuffles them (seeded by `random_seed`) before splitting. The exact
        night_ids used are always saved next to the chains file, so
        postprocessing can replay them exactly regardless of which method
        (if any) produced them -- see `utils/postprocessing_utils.load_input_data`.
    """

    # ----------------------------
    # Load data
    # ----------------------------

    all_files, rejected_nights, night_ids, _ = mixture_model.build_dataset(
        pointing      = pointing,
        tv_subband    = tv_subband,
        abs_directory = ABS_DIR,
    )

    print(f"Rejected {len(rejected_nights)} nights: {rejected_nights}")

    if not all_files:
        # e.g. a sub-band the data doesn't cover (a single coarse channel only
        # reaches one), or every night rejected
        print(f"Nothing to fit for {tv_subband}, pointing {pointing}.")
        return

    if split_method == "chrono":
        all_files, night_ids = mixture_model.chrono_split_nights(all_files, night_ids, half)
        print(f"Chronological split selected. Half {half}: keeping {len(all_files)} nights.")
    elif split_method == "random":
        all_files, night_ids = mixture_model.random_split_nights(all_files, night_ids, half, random_seed)
        print(f"Random split selected. Half {half}: keeping {len(all_files)} nights.")
    elif split_method is not None:
        raise ValueError(f"Unrecognized split_method: {split_method!r}. Expected None, 'chrono', or 'random'.")

    print(f"Remaining: {len(all_files)} nights")

    all_data    = [np.load(f) for f in all_files]
    raw_samples = [d[pol] for d in all_data]
    ssins_flags = [d["ssins"] for d in all_data]
    times       = [d['time_jd'] for d in all_data]
    obsids      = [d['obsids'] for d in all_data]

    with open(ABS_DIR+'data/N_channels.yaml', "r") as f:
        tv_subband_dict = yaml.safe_load(f)

    nbls   = [int(d['nbls'])  for d in all_data] if all('nbls'  in d for d in all_data) else None
    nchans = [int(d['nchan']) for d in all_data] if all('nchan' in d for d in all_data) else None
    sigma_clean = mixture_model.calculate_sigma_clean(raw_samples, tv_subband_dict=tv_subband_dict, tv_subband=tv_subband,
                                                      nbls=nbls, nchans=nchans)
    scale = 1.0
    if rescale_noise is not None:
        raw_samples, sigma_clean, scale = mixture_model.rescale_to_noise(raw_samples, sigma_clean, rescale_noise)
        print(f"Rescaled data by {scale:.6g} to a median noise of {rescale_noise}")
    data_dict = mixture_model.prepare_multiple_nights(raw_samples,
            L=L, flags=ssins_flags, night_ids=night_ids, obsids=obsids, times=times
        )

    # ----------------------------
    # Run model
    # ----------------------------

    model = mixture_model.model
    if non_centred:
        model = reparam(model, config={"beta": LocScaleReparam(centered=0)})

    kernel = NUTS(
        model,
        init_strategy      = init_to_median(num_samples=1000), 
        max_tree_depth     = 8,
        target_accept_prob = 0.6,
    )

    mcmc = MCMC(
        kernel,
        num_warmup  = num_warmup,
        num_samples = num_samples,
        num_chains  = num_chains,
    )

    # random_seed = np.random.randint(0,1000)
    rng_key     = jrandom.key(random_seed)

    mcmc.run(
        rng_key,
        A_pad            = data_dict['A_pad'],
        y_pad            = data_dict['y_pad'],
        mask             = data_dict['mask'],
        y_loc            = data_dict['y_loc'],
        y_scale          = data_dict['y_scale'],
        sigma_clean      = sigma_clean,
        rfi_scale_cutoff = 2 * np.median(sigma_clean),
    )

    timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")

    # Create save directory
    os.makedirs(os.path.join(run_dir, "chains"), exist_ok=True)

    split_tag = f"_half{half}" if split_method is not None else ""
    run_tag   = f"chains_{tv_subband}_p{pointing}_pol{pol}{split_tag}_{timestamp}"

    idata = az.from_numpyro(mcmc)
    idata.to_netcdf(os.path.join(run_dir, "chains", f"{run_tag}.nc"))

    # Save the exact night_ids used for this fit
    with open(os.path.join(run_dir, "chains", f"{run_tag}_nights.txt"), "w") as f:
        f.write("\n".join(night_ids) + "\n")

    if rescale_noise is not None:
        with open(os.path.join(run_dir, "chains", f"{run_tag}_scale.txt"), "w") as f:
            f.write(f"{scale!r}\n")

    notes = f"""
    timestamp     : {timestamp}
    pointing      : {pointing}
    L             : {L}
    polarization  : {pol}
    TV subband    : {tv_subband}
    split_method  : {split_method}
    half          : {half}
    rng_key       : {random_seed}
    reject nights : {rejected_nights}
    num_warmup    : {mcmc.num_warmup}
    num_samples   : {mcmc.num_samples}
    num_chains    : {mcmc.num_chains}
    num_nights    : {len(data_dict['lengths'])}
    data_scale    : {scale}
    non_centred   : {non_centred}
    rfi_dist      : SkewCauchy(loc=loc_rfi, sigma=scale_rfi, a=0.99)
    clean_dist    : Normal(0, sigma_clean)
    fixed         : sigma_clean->calculated from data
    sampled       : loc_rfi, scale_rfi, beta, w_rfi, mu_beta, sigma_beta
    deterministic : w_clean = 1 - w_rfi
    target_accept : 0.6
    notes         : Pathological nights automatically removed from set.
    """

    with open(os.path.join(run_dir, "chains", f"{run_tag}.txt"), "w") as f:
        f.write(notes)

    mcmc.print_summary()


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    job_map = POOLED_JOB_MAP if cli.pool_pointings_requested() else JOB_MAP

    parser = cli.build_parser(
        description="Pipeline stage 2: fit the hierarchical mixture model for one "
                    "(pointing, TV sub-band, polarization) combination.",
        n_jobs=len(job_map),
        epilog="Examples:\n"
               "  python fit.py 0\n"
               "  python fit.py 0 --run-dir out_chrono_split --split-method chrono --half 0\n"
               "  python fit.py 0 --run-dir out_random --split-method random --half 1 --seed 17\n"
               "  python fit.py 4 --pool-pointings          # TV7 xx, all pointings of all nights\n"
               "  python fit.py --n-jobs",
    )
    cli.add_run_dir(parser, default="out",
                    help="run directory to write chains/ into")
    cli.add_half(parser)
    parser.add_argument(
        "--split-method",
        choices=("chrono", "random"),
        default=None,
        help="how to halve the season: 'chrono' splits nights at the midpoint by "
             "time, 'random' shuffles them first (seeded by --seed). Requires --half",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        metavar="INT",
        help=f"random seed for the MCMC rng_key and the random night split "
             f"(default: {RANDOM_SEED}). With --split-method random it is also "
             "appended to --run-dir, e.g. 'out_random' + 17 -> 'out_random_17'",
    )
    cli.add_pool_pointings(parser)
    parser.add_argument("--n-legendre", type=int, default=L, metavar="INT",
                        help=f"number of Legendre background terms (default: {L}, tuned "
                             "for ~30-minute nights; use fewer for short blocks)")
    parser.add_argument("--num-warmup", type=int, default=NUM_WARMUP, metavar="INT",
                        help=f"NUTS warm-up iterations per chain (default: {NUM_WARMUP})")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, metavar="INT",
                        help=f"NUTS samples kept per chain (default: {NUM_SAMPLES})")
    parser.add_argument("--num-chains", type=int, default=N_CHAINS, metavar="INT",
                        help=f"NUTS chains (default: {N_CHAINS}, one per core up to 48)")
    parser.add_argument("--rescale-noise", type=float, default=None, metavar="SIGMA",
                        help="rescale the data so the median noise is SIGMA, keeping the "
                             "priors as wide relative to the noise as in the paper "
                             f"(~{mixture_model.PAPER_SIGMA_NOISE} there); for non-2014 data, "
                             "e.g. MWAX via Birli. Default: no rescaling")
    parser.add_argument("--non-centred", action="store_true",
                        help="sample the Legendre coefficients non-centred (same model, "
                             "no funnel); recommended when fitting only a few nights")
    args = cli.parse_args(parser)

    # A half-split is only defined by both halves of the description, so
    # accept either both flags or neither rather than silently ignoring one.
    if (args.split_method is None) != (args.half is None):
        parser.error("--split-method and --half must be given together "
                     "(omit both for a full-season fit)")

    run_dir = args.run_dir
    if args.split_method == "random":
        run_dir = f"{run_dir}_{args.seed}"
    run_dir = cli.resolve_run_dir(run_dir, ABS_DIR)

    _, tv_subband, pointing, pol = job_map[args.job_index]
    L = args.n_legendre

    cli.print_config(
        "Fit mixture model",
        job_index    = args.job_index,
        pointing     = pointing,
        L            = L,
        polarization = pol,
        tv_subband   = tv_subband,
        split_method = args.split_method,
        half         = args.half,
        seed         = args.seed,
        run_dir      = run_dir,
    )

    # Run the mixture model
    run_mixture_model(pointing, L, pol, tv_subband, args.seed, run_dir,
                        split_method=args.split_method, half=args.half,
                        num_warmup=args.num_warmup, num_samples=args.num_samples,
                        num_chains=args.num_chains, rescale_noise=args.rescale_noise,
                        non_centred=args.non_centred)
