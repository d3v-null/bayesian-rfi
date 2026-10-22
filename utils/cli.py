"""
Shared command-line interface for the top-level pipeline scripts.

Every stage of the pipeline runs as a SLURM array job, so every top-level
script takes exactly the same first argument -- `job_index`, the index of this
task inside that stage's job grid (i.e. `$SLURM_ARRAY_TASK_ID`) -- followed by
optional, stage-specific flags. Nothing else is positional, so the stages stay
interchangeable from a caller's point of view and the `.sh` wrappers stay thin
pass-throughs.

The grid itself is always defined in Python (a module-level `JOB_MAP` or
`N_JOBS`). `--n-jobs` prints its size, which is the
one number the `#SBATCH --array=` bound has to agree with:

    python fit.py --n-jobs        # -> 50, i.e. #SBATCH --array=0-49

Stages that aren't parallel over anything (`summarize_season.py`) still take a
`job_index`, which must then be 0.
"""

import argparse
import os

# Set by build_parser() on the parser itself, so parse_args() can recover the
# grid size without every caller having to pass it twice.
_N_JOBS_DEST = "_n_jobs"


def build_parser(description, n_jobs, epilog=None):
    """
    Build an ArgumentParser carrying the CLI contract shared by every stage.

    Adds the `job_index` positional (validated against `n_jobs` in
    `parse_args`) and the `--n-jobs` introspection flag. Callers add their own
    stage-specific options to the returned parser, then call `parse_args`.

    Parameters
    ----------
    description : str
        One-line description of the stage, shown by `--help`.
    n_jobs : int
        Number of tasks in this stage's job grid (`len(JOB_MAP)`, `N_JOBS`, ...).
    epilog : str, optional
        Extra `--help` text, e.g. usage examples. Printed verbatim.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(**{_N_JOBS_DEST: n_jobs})

    # nargs="?" so `--n-jobs` can be used on its own; parse_args() enforces
    # that job_index is present otherwise.
    parser.add_argument(
        "job_index",
        nargs="?",
        type=int,
        help=f"index of this task in the stage's job grid, 0 to {n_jobs - 1} "
             "(pass $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--n-jobs",
        action="store_true",
        dest="print_n_jobs",
        help="print the size of this stage's job grid (the #SBATCH --array bound) and exit",
    )
    return parser


def parse_args(parser, argv=None):
    """
    Parse and validate arguments from a parser built by `build_parser`.

    Handles `--n-jobs` (prints the grid size and exits 0) and range-checks
    `job_index`, exiting 2 with a usage message if it's missing or out of
    range -- checked here rather than in each stage so the failure mode is
    identical everywhere.

    Returns
    -------
    argparse.Namespace
        With `job_index` guaranteed to be a valid index into the grid.
    """
    args = parser.parse_args(argv)
    n_jobs = parser.get_default(_N_JOBS_DEST)

    if args.print_n_jobs:
        print(n_jobs)
        raise SystemExit(0)

    if args.job_index is None:
        parser.error("job_index is required (pass $SLURM_ARRAY_TASK_ID)")

    if not 0 <= args.job_index < n_jobs:
        parser.error(f"job_index must be in [0, {n_jobs}), got {args.job_index}")

    return args


def add_run_dir(parser, default, help=None):
    """
    Add the `--run-dir` option shared by every stage that reads or writes a
    run's output tree (`{run_dir}/chains/`, `{run_dir}/processed/`, ...).

    The value may be absolute, or relative to the project root -- see
    `resolve_run_dir`.
    """
    parser.add_argument(
        "--run-dir",
        default=default,
        metavar="DIR",
        help=(help or "run output directory, absolute or relative to the "
                      "project root") + f" (default: {default})",
    )


def add_half(parser):
    """
    Add the `--half` option shared by the stages that support half-season
    splits (fit, postprocess, summarize_season). Omitted means the full season.
    """
    parser.add_argument(
        "--half",
        type=int,
        choices=(0, 1),
        default=None,
        help="restrict to half the season (0 or 1); omit for a full-season run",
    )


def resolve_run_dir(run_dir, abs_dir):
    """
    Resolve a `--run-dir` value to an absolute path.

    Absolute paths are used as given; anything else is interpreted relative to
    the project root, so `--run-dir out_chrono_split` and
    `--run-dir /oscar/.../BayesRFI/out_chrono_split/` mean the same thing.
    """
    if os.path.isabs(run_dir):
        return run_dir
    return os.path.join(abs_dir, run_dir)


def print_config(title, **fields):
    """
    Print the banner every stage opens with: a title, then one `key : value`
    line per field, aligned. Field order is the order the keyword arguments
    were passed in.
    """
    print("\n----------------------")
    print(title.upper())
    print("----------------------\n")
    width = max((len(k) for k in fields), default=0)
    for key, value in fields.items():
        print(f"{key.ljust(width)} : {value}")
    print()


def add_pool_pointings(parser):
    """
    Add the `--pool-pointings` option shared by the stages whose job grid is
    over (pointing, TV sub-band, polarization) (fit, postprocess).

    With it, every pointing's (night, pointing) blocks go into one fit per
    (TV sub-band, polarization), which is how the observations of a single
    night are stacked. The grid then has no pointing axis, so call
    `pool_pointings_requested` *before* `build_parser` to size it.
    """
    parser.add_argument(
        "--pool-pointings",
        action="store_true",
        help="fit every pointing's (night, pointing) blocks together, one task per "
             "(TV sub-band, polarization); use this to stack a single night",
    )


def pool_pointings_requested(argv=None):
    """
    Whether `--pool-pointings` is on the command line, read ahead of the full
    parse because it changes the size of the job grid `build_parser` needs.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--pool-pointings", action="store_true")
    known, _ = pre.parse_known_args(argv)
    return known.pool_pointings
