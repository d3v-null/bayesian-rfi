# bayesian-rfi

A pipeline for statistically characterizing RFI in MWA (Murchison Widefield Array) radio-astronomy visibilities.

Raw correlator FITS files are reduced with [SSINS](https://github.com/mwilensky768/SSINS), then [numpyro](https://num.pyro.ai/en/stable/) is used to fit a hierarchical Bayesian mixture model per
(pointing, TV sub-band, polarization) combination to probabilistically classify every time-sample as
"clean" vs. "RFI"-contaminated. Results are aggregated into a single HDF5 file, and the
resulting soft labels (continuous RFI probability) and their Bayesian error bars are
propagated back into [pyuvdata](https://pyuvdata.readthedocs.io/en/latest/) `UVFlag` objects (metric mode) for use downstream.

## Repository layout

| Path | What it is |
| --- | --- |
| `preprocess.py` | Stage 1 — SSINS reduction of raw visibilities |
| `fit.py` | Stage 2 — fit the mixture model |
| `postprocess.py` | Stage 3 — chain diagnostics, classification, results store |
| `summarize_season.py` | Stage 4 — season-level aggregation, summary plots |
| `propagate.py` | Stage 5 — write labels back into `UVFlag` files |
| `mixture_model.py` | The NumPyro model, dataset construction, night selection |
| `utils/` | Shared helpers: `cli`, `hdf5_utils`, `mm_utils`, `preprocessing_utils`, `postprocessing_utils`, `plotting_utils` |

## Environment

```bash
pip install -r requirements.txt
```

Key dependencies: `pyuvdata` (3.x), `SSINS` (git, pyuvdata-3 compatible commit pinned in
`requirements.txt`), `numpyro`, `jax`, `arviz` (0.x API), `h5py`, `astropy`, `pandas`,
`matplotlib`/`seaborn`/`corner`.

### Paths

All stages locate their input and output trees through two environment variables (the hard-coded
defaults are the authors' cluster paths):

| Variable | Meaning | Used by |
| --- | --- | --- |
| `BAYESRFI_DATA_DIR` | directory holding the raw `.fits` / `.metafits` files (flat, one directory) | `preprocess.py` |
| `BAYESRFI_ABS_DIR` | project root: stage 1 writes `data/` here, later stages read it and write `<run-dir>/` under it | all stages |
| `BAYESRFI_NPROC` | number of observations `preprocess.py` reads in parallel (default 1; each needs ~30 GB) | `preprocess.py` |

```bash
export BAYESRFI_DATA_DIR=/path/to/raw/
export BAYESRFI_ABS_DIR=/path/to/project/
```

## Command-line convention

Each stage is meant to run as one task of a SLURM job array, so every stage takes the **same first
argument**: `job_index`, the index of this task within that stage's job grid (under SLURM, pass
`$SLURM_ARRAY_TASK_ID`). Everything else is an optional flag. The shared part of the interface
lives in `utils/cli.py`.

```bash
python <stage>.py <job_index> [--run-dir DIR] [--half {0,1}] [stage-specific flags]
python <stage>.py --n-jobs        # size of this stage's job grid; nothing else runs
python <stage>.py --help          # full options and examples for that stage
```

* `--run-dir` sets the output file path. Defaults to `out`.
* `--n-jobs` corresponds to the total number of (pointing, TV sub-band, polarization) combinations

  ```bash
  python fit.py --n-jobs        # -> 50, so e.g. #SBATCH --array=0-49
  ```

* `summarize_season.py` isn't parallel over anything, so its grid has exactly one task and its
  only valid `job_index` is `0`. It takes the argument anyway, for consistency

The batch submission scripts are specific to the cluster this was developed on and are not part
of this repository, but a wrapper for any stage is a one-liner; e.g.

```bash
#!/bin/bash
#SBATCH --array=0-49            # must equal `python fit.py --n-jobs`
python fit.py "$SLURM_ARRAY_TASK_ID" --run-dir out
```

## Running the pipeline

Stages are listed in execution order. Each one consumes the previous one's output.

### 1. Preprocess raw visibilities — `preprocess.py`

```bash
python preprocess.py <job_index>          # 0 .. N, one per night/pointing
```

One task per (night, pointing). Groups raw `.fits` observations into (night, pointing) sets,
builds an `SS`/`INS` object per obsid, runs an SSINS match filter tuned for DTV shapes
(`TV6`–`TV9`) plus narrowband/streak RFI, seeds an initial `UVFlag` with coarse-band and
edge-time flags, and writes per-sub-band frequency-averaged metric time series (`.npz`) plus
one `UVFlag` file per obsid.

Set `BAYESRFI_DATA_DIR` and `BAYESRFI_ABS_DIR` (see *Paths* above). Reading one observation takes
several minutes and ~30 GB; `BAYESRFI_NPROC=N` reads N observations at once. Each observation's INS is
cached under `{OUT_DIR}/ins/` (INS objects cannot be pickled between processes, and the cache lets a
night be resumed after a crash). The coarse-band mask is derived from the frequency axis, so any
fine-channel width works, and the number of baselines averaged is stored per night for the noise model.

**Out:** `{OUT_DIR}/tv_split/{tv_subband}/processed_{obsn}_p{pointing}.npz`,
`{OUT_DIR}/uvflag/{obsid}_flags.h5`, `{OUT_DIR}/ins/{obsid}_SSINS_{data,mask}.h5`, `{OUT_DIR}/N_channels.yaml`

### 2. Fit the mixture model — `fit.py`

```bash
python fit.py <job_index>                                          # full season, into default out/ directory
python fit.py <job_index> --run-dir out_chrono_split \
                          --split-method chrono --half 0           # chronological half-split
python fit.py <job_index> --run-dir out_random \
                          --split-method random --half 1 --seed 17 # random half-split
```

One task per (pointing, TV sub-band, polarization). Loads every night's `.npz` for that
combination (dropping nights that exceed the SSINS flagged-fraction tolerance), fits the
hierarchical mixture model with NUTS across 48 chains (fewer on machines with fewer cores), and writes the posterior plus a notes
sidecar. The nights actually used are written verbatim to a `..._nights.txt` sidecar so stage 3
can replay exactly the same set (e.g. in case of random split)

The option to run a random or chronological split of the data is included for model testing/validation
purposes.

With `--split-method random`, the seed is appended to the run directory name
(`out_random` + seed 17 → `out_random_17`), so different seeds never collide.

**Out:** `{run_dir}/chains/chains_{tv}_p{pointing}_pol{pol}[_half{N}]_{timestamp}.{nc,txt}` and
`..._nights.txt`

### 3. Postprocess the chains — `postprocess.py`

```bash
python postprocess.py <job_index>                                  # full season, from default out/ directory
python postprocess.py <job_index> --run-dir out_chrono_split --half 0
```

Same job grid as stage 2. Loads the most recent matching chains file, drops "stuck" chains,
recomputes convergence diagnostics, reconstructs per-night backgrounds/residuals, classifies
every sample (soft + hard labels), computes goodness-of-fit, writes diagnostic plots, and
appends everything to the shared results store behind a file lock (safe for concurrent array
tasks). The full per-draw soft labels are far too big for that shared file and go to their own
per-run HDF5 file.

**Out:** `{run_dir}/processed/season_results.h5` (or `half{N}_season_results.h5`),
`{run_dir}/processed/processed_{...}/` plots, `{run_dir}/processed/soft_label_samples/`

### 4. Season-level statistics — `summarize_season.py`

```bash
python summarize_season.py 0                           # full season, out/
python summarize_season.py 0 --run-dir out_chrono_split --half 0
```

Reads the whole `season_results.h5` into one
tidy dataframe and produces season-level plots: heatmaps of `w_rfi`/`loc_rfi`/`scale_rfi` vs.
pointing/sub-band, per-night reduced chi-squared, posterior-vs-empirical RFI fraction, and the RFI
responsibility time series.

**Out:** `{run_dir}/processed/[half{N}_]seasonal_statistics/`

### 5. Propagate labels into the UVFlag files — `propagate.py`

```bash
python propagate.py <job_index>                        # default: reads out/processed/season_results.h5
```

Each task owns a disjoint chunk of the `UVFlag` files.
Converts each file to metric mode and writes the continuous soft label (posterior mean RFI
probability) into `metric_array` and its Bayesian error bar (posterior std) into
`weights_array`. Nights that were rejected from the fit have no posterior to look up at all, so
their sub-bands are force-set to metric=1.0 / weights=0.0.

Note that this is all experimental and subject to change.

**Out:** `data*/uvflag/{obsid}_flags.h5`, rewritten in place

## Half-season splits

Stages 2–4 accept a half-split for reproducibility checks: fit each half of the season
separately and compare posteriors. `--split-method chrono` sorts nights by time and splits at
the midpoint; `--split-method random` shuffles them first (seeded by `--seed`). A full run of
one half looks like:

```bash
python fit.py         <job_index> --run-dir out_chrono_split --split-method chrono --half 0
python postprocess.py <job_index> --run-dir out_chrono_split --half 0
python summarize_season.py 0      --run-dir out_chrono_split --half 0
```

Half-split runs write `half{N}_season_results.h5` alongside (not on top of) the full-season
store, and their chains are tagged `_half{N}_` in the filename so the two kinds are never mixed.

## Debugging a single task

Nothing about a stage requires a scheduler: run it directly with the task index it would have
received, and it does exactly that task.

```bash
python postprocess.py 7 --run-dir out_chrono_split --half 0
```

`--help` on any stage prints its options and worked examples; `--n-jobs` prints how many tasks
that stage's grid has.
