"""
Write a small synthetic observing night for an end-to-end test of the pipeline.

Mimics what Birli writes for ONE MWAX coarse channel per observation: one
`{obsid}_ch{N}.uvfits` per 2-minute observation (2 s integrations, 40 kHz fine
channels, 4 pols) plus a minimal `{obsid}.metafits` carrying just the header
keys `mm_utils.sort_season` reads (GRIDNUM, DATE-OBS). Visibilities are a
constant sky term plus thermal noise, scaled by a smooth night-long gain drift
(the background the Legendre model fits). Broadband RFI bursts -- noise-like,
like DTV -- are injected at known times; their diff-sample times are written to
`truth.json` so the posterior soft labels can be scored.

    python tests/make_synthetic_night.py OUT_DIR [--seed 1]

Then, with BAYESRFI_DATA_DIR=OUT_DIR/ (trailing slash):

    python preprocess.py <i> --input uvfits     # for each i < --n-jobs
"""

import argparse
import json
import os

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.time import Time
from pyuvdata import Telescope, UVData

MWA_LOC      = EarthLocation(lat=-26.703319, lon=116.67081, height=377.83)
COARSE_CHAN  = 144             # receiver channel: centre 184.32 MHz, inside TV7 (181-188 MHz)
N_FINE       = 32              # 40 kHz fine channels
N_TIMES      = 60              # 2 s integrations per 2-minute observation
INT_TIME_S   = 2.0
N_ANTS       = 16
NIGHT_START  = "2026-09-05T13:00:00"

# (pointing GRIDNUM, number of observations), observed back to back
POINTING_BLOCKS = [(0, 3), (2, 3)]

# (observation index within the night, first, last timestep, sigma_rfi / sigma_noise, pol scale (xx, yy))
BURSTS = [
    (0, 20, 23, 0.6,  (1.0, 0.5)),
    (1, 40, 40, 1.0,  (1.0, 0.5)),   # one-integration "blip"
    (2, 10, 18, 0.35, (1.0, 0.5)),   # longer, fainter event
    (4, 30, 33, 0.6,  (1.0, 0.5)),
    (5, 50, 51, 0.8,  (1.0, 0.5)),
]


def gain(t_s, t0_s):
    """Smooth, percent-level receiver gain drift over the night."""
    x = (t_s - t0_s) / 3600.0
    return 1.0 + 0.02 * np.sin(2 * np.pi * x / 1.3) + 0.01 * x


def make_obs(obsid, pointing, obs_idx, t0_gps, rng, out_dir):
    freqs = COARSE_CHAN * 1.28e6 - 0.64e6 + 20e3 + 40e3 * np.arange(N_FINE)

    enu = rng.uniform(-150, 150, size=(N_ANTS, 3))
    enu[:, 2] = 0.0
    telescope = Telescope.new(
        name="MWA", location=MWA_LOC,
        antenna_positions={f"Tile{a:03d}": enu[a] for a in range(N_ANTS)},
        antenna_numbers=np.arange(N_ANTS), instrument="MWA",
        x_orientation="east", feeds=["x", "y"], mount_type="phased",
        update_from_known=False,
    )
    antpairs = [(a, b) for a in range(N_ANTS) for b in range(a, N_ANTS)]
    t_gps    = obsid + INT_TIME_S * (np.arange(N_TIMES) + 0.5)
    times    = Time(t_gps, format="gps").jd

    uvd = UVData.new(
        freq_array=freqs, polarization_array=np.array([-5, -6, -7, -8]),
        times=times, telescope=telescope, antpairs=antpairs, do_blt_outer=True,
        integration_time=INT_TIME_S, channel_width=40e3, empty=True,
    )
    nbls  = len(antpairs)
    shape = (N_TIMES, nbls, N_FINE, 4)

    sky   = 5.0 * np.exp(2j * np.pi * rng.uniform(size=(1, nbls, 1, 1)))
    noise = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
    vis   = sky + noise

    truth = []
    for b_obs, first, last, amp, pol_scale in BURSTS:
        if b_obs != obs_idx:
            continue
        n = last - first + 1
        rfi = (rng.normal(size=(n, nbls, N_FINE, 4)) + 1j * rng.normal(size=(n, nbls, N_FINE, 4))) / np.sqrt(2)
        scale = np.array([pol_scale[0], pol_scale[1], 0.7, 0.7])
        vis[first:last + 1] += amp * scale * rfi
        # time-differenced samples (k, k+1) touching the burst, as SSINS diff times
        k = np.arange(max(first - 1, 0), min(last, N_TIMES - 2) + 1)
        truth.extend(Time((t_gps[k] + t_gps[k + 1]) / 2, format="gps").jd.tolist())

    vis *= gain(t_gps, t0_gps)[:, None, None, None]
    # UVData.new with do_blt_outer orders baseline-times time-major
    uvd.data_array    = vis.reshape(N_TIMES * nbls, N_FINE, 4).astype(np.complex64)
    uvd.flag_array[:] = False
    uvd.nsample_array[:] = 1.0
    autos = uvd.ant_1_array == uvd.ant_2_array
    uvd.data_array[autos] = np.abs(uvd.data_array[autos])
    uvd.vis_units = "uncalib"
    uvd.history = "synthetic night for bayesian-rfi tests"

    uvd.write_uvfits(os.path.join(out_dir, f"{obsid}_ch{COARSE_CHAN}.uvfits"), force_phase=True)

    hdr = fits.Header()
    hdr["GPSTIME"]  = obsid
    hdr["DATE-OBS"] = Time(obsid, format="gps").utc.isot[:19]
    hdr["GRIDNUM"]  = pointing
    hdr["GRIDNAME"] = "sweet"
    hdr["CHANNELS"] = str(COARSE_CHAN)
    hdr["INTTIME"]  = INT_TIME_S
    fits.PrimaryHDU(header=hdr).writeto(os.path.join(out_dir, f"{obsid}.metafits"), overwrite=True)
    return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng    = np.random.default_rng(args.seed)
    t0_gps = int(Time(NIGHT_START, scale="utc").gps)

    truth, obsid, obs_idx = {}, t0_gps, 0
    for pointing, n_obs in POINTING_BLOCKS:
        for _ in range(n_obs):
            truth[str(obsid)] = make_obs(obsid, pointing, obs_idx, t0_gps, rng, args.out_dir)
            print(f"wrote obsid {obsid} pointing {pointing} ({len(truth[str(obsid)])} RFI diff samples)")
            obsid += 120 + 8   # 2-minute observations, 8 s apart
            obs_idx += 1

    with open(os.path.join(args.out_dir, "truth.json"), "w") as f:
        json.dump({"rfi_times_jd": truth, "bursts": BURSTS}, f, indent=1)


if __name__ == "__main__":
    main()
