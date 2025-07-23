import numpy as np
from numba import jit

def denoise_events(events, tau_d=10000.0, tau_n=200.0, sigma_n=1.68, delta_d=0.05):
    """Fast Numba-accelerated rolling-window denoiser."""
    if not events:
        return []

    times = np.array([e['t'] for e in events], dtype=np.float64)
    xs = np.array([e['x'] for e in events], dtype=np.int32)
    ys = np.array([e['y'] for e in events], dtype=np.int32)

    sort_idx = np.argsort(times)
    times = times[sort_idx]
    xs = xs[sort_idx]
    ys = ys[sort_idx]

    keep_mask = _fast_denoise_numba(times, xs, ys, tau_d, tau_n, sigma_n, delta_d)

    # Restore original order
    unsorted_keep = np.zeros_like(keep_mask)
    unsorted_keep[sort_idx] = keep_mask

    return [e for i, e in enumerate(events) if unsorted_keep[i]]


@jit(nopython=True)
def _fast_denoise_numba(times, xs, ys, tau_d, tau_n, sigma_n, delta_d):
    n = len(times)
    keep = np.zeros(n, dtype=np.bool_)

    max_back = 10000  # Max number of events to look back for speed

    for i in range(n):
        t_i = times[i]
        x_i = xs[i]
        y_i = ys[i]

        D = 0.0

        # Only consider recent events within tau_d (bounded by max_back)
        for j in range(i - 1, max(-1, i - max_back), -1):
            dt = t_i - times[j]
            if dt > tau_d:
                break  # Too far back in time
            dx = xs[j] - x_i
            dy = ys[j] - y_i
            if abs(dx) > 1 or abs(dy) > 1:
                continue

            temp_w = np.exp(-dt / tau_n)
            spat_w = np.exp(-(dx**2 + dy**2) / (2 * sigma_n**2))
            D += temp_w * spat_w

        keep[i] = D >= delta_d

    return keep
