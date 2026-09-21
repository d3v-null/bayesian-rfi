import glob
import re
from collections import defaultdict

import numpy as np
from astropy.io import fits


def pad_time_array(t):
    """
    Fill gaps in a time array with evenly-spaced interpolated timestamps.

    Detects gaps larger than one integration (based on the median cadence) and
    inserts synthetic timestamps to produce a uniformly-spaced grid. Used to
    align SSINS metrics to a consistent time axis across concatenated observations.

    Parameters
    ----------
    t : array-like
        Observed timestamps in JD. Need not be gapless.

    Returns
    -------
    t_filled : ndarray
        Uniformly-spaced time array covering the same range as t, with gaps
        filled by interpolated timestamps.
    """
    t = np.asarray(t, dtype=float)

    dt = np.diff(t)
    dt0 = np.median(dt)

    t_filled = [t[0]]

    for i in range(len(dt)):
        gap = dt[i]

        # how many dt0-sized intervals fit in this gap
        n_intervals = int(np.round(gap / dt0))

        # if normal step, just append next time
        if n_intervals <= 1:
            t_filled.append(t[i+1])
            continue

        # fill missing points
        new_ts = t[i] + dt0 * np.arange(1, n_intervals)
        t_filled.extend(new_ts)
        t_filled.append(t[i+1])

    return np.array(t_filled)


def pad_ssins(metric, t_filled, t_orig):
    """
    Align a metric array to a padded time grid, inserting NaN for missing times.

    For each timestamp in t_filled, copies the corresponding value from metric
    if that timestamp exists in t_orig, otherwise inserts NaN.

    Parameters
    ----------
    metric : ndarray, shape (Ntimes, ...)
        Metric values at the original observed timestamps.
    t_filled : ndarray
        Uniformly-spaced time grid from pad_time_array.
    t_orig : ndarray
        Original observed timestamps, matching the first axis of metric.

    Returns
    -------
    padded_metric : ndarray, shape (len(t_filled), ...)
        Metric aligned to t_filled, with NaN inserted for unobserved times.
    """
    
    padded_metric = []
    for pt in t_filled:

        if pt in t_orig:
            tdx = np.where(t_orig == pt)[0]
            padded_metric.extend(metric[tdx])

        else:
            padded_metric.append(np.nan)

    return np.asarray(padded_metric)


def sort_season(datapath):
    """
    Group observations from a season by (date, pointing) from their metafits headers.

    Globs for all .metafits files under datapath, extracts the obsid, pointing
    (GRIDNUM), and date (DATE-OBS) from each header, and returns a dict mapping
    (date, pointing) keys to lists of obsids.

    Parameters
    ----------
    datapath : str
        Directory containing .metafits files. A trailing slash is not required.

    Returns
    -------
    night_pointings : defaultdict(list)
        Maps (date, pointing) -> [obsid, ...], where date is 'YYYY-MM-DD' and
        pointing is the integer GRIDNUM from the metafits header.
    """

    meta = glob.glob(datapath+"*.metafits")
    obsids = [re.search(r'\d{10}', m).group() for m in meta]

    # --- Extract pointing and night for each obsid ---
    obs_info = {}
    for obsid, m in zip(obsids, meta):
        with fits.open(m) as hdul:
            header = hdul[0].header
            pointing = header['GRIDNUM']
            # Note: DATE-OBS is UTC; MWA nights fall within 10:00–22:00 UTC so
            # no night straddles a UTC date boundary.
            date = header['DATE-OBS'][:10]   # "YYYY-MM-DD", strip time
        obs_info[obsid] = {'pointing': pointing, 'date': date}

    # --- Group by (night, pointing) ---
    night_pointings = defaultdict(list)
    for obsid, info in obs_info.items():
        key = (info['date'], info['pointing'])
        night_pointings[key].append(obsid)

    return night_pointings
