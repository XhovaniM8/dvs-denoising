import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def events_to_frames(
    events,
    width=None,
    height=None,
    dt=None,
    n_frames=None,
    events_per_frame=None,
    signed=True,
    normalize=True,
):
    """
    Convert event list -> 3D array (frames, H, W).
    Provide exactly one of: dt, n_frames, events_per_frame.
    """
    if not events:
        return np.zeros((0, 0, 0), dtype=np.float32)

    # Extract arrays
    t = np.asarray([e['t'] for e in events], dtype=np.float64)
    x = np.asarray([e['x'] for e in events], dtype=np.int32)
    y = np.asarray([e['y'] for e in events], dtype=np.int32)
    p = np.asarray([e['p'] for e in events], dtype=np.int8)

    # Infer sensor size
    if width is None:  width = int(x.max()) + 1
    if height is None: height = int(y.max()) + 1

    # Sort by time
    order = np.argsort(t)
    t, x, y, p = t[order], x[order], y[order], p[order]

    t0, t1 = t[0], t[-1]
    total_events = len(t)

    # Determine bin edges
    if dt is not None:
        edges = np.arange(t0, t1 + dt, dt)
    elif n_frames is not None:
        edges = np.linspace(t0, t1, n_frames + 1)
    elif events_per_frame is not None:
        # Build edges so each frame has ~same #events
        idx_edges = np.arange(0, total_events + events_per_frame, events_per_frame)
        idx_edges[-1] = total_events
        # Convert to time edges
        edges = np.concatenate(([t0], t[idx_edges[1:-1]], [t1]))
    else:
        raise ValueError("Provide dt, n_frames, or events_per_frame.")

    nF = len(edges) - 1
    frames = np.zeros((nF, height, width), dtype=np.float32)

    # Digitize events into bins
    bin_ids = np.searchsorted(edges, t, side="right") - 1
    # Clip possible last-edge overflow
    bin_ids = np.clip(bin_ids, 0, nF - 1)

    # Accumulate
    if signed:
        vals = np.where(p > 0, 1.0, -1.0)
    else:
        vals = np.ones_like(p, dtype=np.float32)

    # Vectorized scatter-add per frame
    for b in range(nF):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        xb, yb, vb = x[mask], y[mask], vals[mask]
        np.add.at(frames[b], (yb, xb), vb)

    if normalize:
        # Normalize per frame to [0,1] for display (signed -> map -max..+max)
        for i in range(nF):
            f = frames[i]
            if signed:
                m = np.max(np.abs(f))
                if m > 0:
                    frames[i] = 0.5 + 0.5 * (f / m)  # center 0 at 0.5
            else:
                m = f.max()
                if m > 0:
                    frames[i] = f / m

    return frames, edges
