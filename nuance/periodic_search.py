"""
The periodic search module provides functions to compute the probability of 
a periodic signal to be present in the data, using quantities computed from single 
events statistics.
"""

import multiprocess as mp
from functools import partial
import jax
import numpy as np
from tqdm.auto import tqdm

from nuance import DEVICES_COUNT, core
from nuance.utils import interp_split_times


def periodic_search(epochs, durations, ls, snr_f, progress=True):
    """Returns a function that performs the periodic search given an array of periods.

    Parameters
    ----------
    epochs : np.ndarray
        Values of epochs for with the likelihoods have been computed.
    durations : np.ndarray
        Values of durations for with the likelihoods have been computed.
    ls : list, tuple, np.ndarray
        Tuple of (likelihoods, depth, depths variance).
    snr_f : callable
        Function that computes the SNR given the epoch, duration and period.
    progress : bool, optional
        wether to show progress bar, by default True

    Returns
    -------
    callable
        Function that computes the SNR and parameters for each period.
    """

    fold_f = _fold_ll(epochs, *ls)

    def _progress(x, **kwargs):
        return tqdm(x, **kwargs) if progress else x

    def function(periods, duration_min=None, duration_max=None, processes=DEVICES_COUNT, batch_size=DEVICES_COUNT):

        if duration_min is None:
            duration_min = np.full_like(periods, np.amin(durations))

        if duration_max is None:
            duration_max = np.full_like(periods, np.amax(durations))

        # Use multiprocessing to get the optimal epoch and duration at each period.
        solve_f = partial(_solve, fold_f)
        ctx = mp.get_context('spawn')  # Can't use fork with jax.
        with ctx.Pool(processes=processes) as pool:

            params = np.zeros((len(periods), 3))
            chunks = [(periods[i::processes], durations, duration_min[i::processes], duration_max[i::processes]) for i in range(processes)]
            for i, result in enumerate(_progress(pool.starmap(solve_f, chunks), total=processes)):
                epochs_chunk, durations_chunk, periods_chunk = result
                params[i::processes, 0] = epochs_chunk
                params[i::processes, 1] = durations_chunk
                params[i::processes, 2] = periods_chunk

            pool.close()
            pool.join()

        # Use jax.pmap to get the SNR at each period.
        snr = np.zeros(len(periods))  # Output array.
        snr_jax = jax.pmap(snr_f, in_axes=(0, 0, 0))
        for i in _progress(range(0, len(periods), batch_size), unit_scale=batch_size):
            imin = i
            imax = i + batch_size
            snr[imin:imax] = snr_jax(params[imin:imax, 0], params[imin:imax, 1], params[imin:imax, 2])

        return snr, params

    return function


def _fold_ll(epochs, lls, z, vz):

    f_idx = core.nearest_neighbors_idx(epochs)

    def _fold(times):
        idx = f_idx(times)
        lls_idx = lls[idx]
        z_idx = z[idx]
        vz_idx = vz[idx]

        vZ = 1 / np.sum(1 / vz_idx, 0)
        Z = vZ * np.sum(z_idx / vz_idx, 0)
        P1 = np.sum(lls_idx, 0)
        P2 = 0.5 * np.sum(
            np.log(vz_idx) - np.log(vz_idx + vZ) + (z_idx - Z) ** 2 / (vz_idx + vZ), 0
        )

        return P1 - P2

    def fun(period):
        times = interp_split_times(epochs, period)
        phase = times[0] / period
        lls = _fold(times)
        return phase, lls

    return fun


def _solve(fold_f, periods, durations, duration_min, duration_max):

    epoch_vals = np.zeros_like(periods)
    duration_vals = np.zeros_like(periods)

    for i, period in enumerate(periods):
        phase, lls = fold_f(period)

        # Enforce duration limits.
        lls = np.where(durations < duration_min[i], np.nan, lls)
        lls = np.where(durations > duration_max[i], np.nan, lls)

        # Find the peak at this period.
        epoch_i, duration_i = np.unravel_index(np.nanargmax(lls), lls.shape)

        # Save the epoch and duration of the peak.
        epoch_vals[i] = phase[epoch_i] * period
        duration_vals[i] = durations[duration_i]

    return epoch_vals, duration_vals, periods


def main():
    return


if __name__ == '__main__':
    main()
