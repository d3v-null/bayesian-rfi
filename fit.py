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
from numpyro.infer import MCMC, NUTS, init_to_median

import mixture_model

numpyro.set_host_device_count(48)  # assumes 48-core device

from jax import random as jrandom

ABS_DIR       = "/oscar/data/jpober/jmduchar/Research/BayesRFI/"
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

def run_mixture_model(pointing, L, pol, tv_subband, random_seed, run_dir,
                    split_method=None, half=None):
    """
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

    sigma_clean = mixture_model.calculate_sigma_clean(raw_samples, tv_subband_dict=tv_subband_dict, tv_subband=tv_subband)
    data_dict = mixture_model.prepare_multiple_nights(raw_samples,
            L=L, flags=ssins_flags, night_ids=night_ids, obsids=obsids, times=times
        )

    # ----------------------------
    # Run model
    # ----------------------------

    kernel = NUTS(
        mixture_model.model, 
        init_strategy      = init_to_median(num_samples=1000), 
        max_tree_depth     = 8,
        target_accept_prob = 0.6,
    )

    mcmc = MCMC(
        kernel,
        num_warmup  = 6000,
        num_samples = 3000,
        num_chains  = 48,
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

    parser = cli.build_parser(
        description="Pipeline stage 2: fit the hierarchical mixture model for one "
                    "(pointing, TV sub-band, polarization) combination.",
        n_jobs=len(JOB_MAP),
        epilog="Examples:\n"
               "  python fit.py 0\n"
               "  python fit.py 0 --run-dir out_chrono_split --split-method chrono --half 0\n"
               "  python fit.py 0 --run-dir out_random --split-method random --half 1 --seed 17\n"
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

    L, tv_subband, pointing, pol = JOB_MAP[args.job_index]

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
                        split_method=args.split_method, half=args.half)
