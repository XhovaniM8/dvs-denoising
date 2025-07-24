"""
convert_events_to_frames.py

Convert a list of DVS events into a sequence of grayscale frames for visualization or video export.
Supports various binning strategies including fixed time duration, fixed number of frames,
or fixed number of events per frame.

Each output frame accumulates signed (ON/OFF) or unsigned event polarities into pixel intensities,
with optional normalization for display or storage.

Author: Xhovani Mali
"""

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
    Convert a list of events to a 3D array of frames: (n_frames, height, width).

    Parameters:
    - events: list of dicts with keys {'x', 'y', 't', 'p'}
    - width, height: spatial dimensions (auto-inferred if not provided)
    - dt: duration of each frame in microseconds
    - n_frames: number of output frames
    - events_per_frame: number of events per frame
    - signed: if True, use polarity (1/-1); else accumulate as binary
    - normalize: if True, normalize frame intensities to [0, 1]

    Returns:
    - frames: np.ndarray of shape (n_frames, height, width)
    - edges: list of frame boundary timestamps
    """
    if not events:
        return np.zeros((0, 0, 0), dtype=np.float32), []

    # Extract arrays
    t = np.asarray([e['t'] for e in events], dtype=np.float64)
    x = np.asarray([e['x'] for e in events], dtype=np.int32)
    y = np.asarray([e['y'] for e in events], dtype=np.int32)
    p = np.asarray([e['p'] for e in events], dtype=np.int8)

    # Infer sensor size
    if width is None:
        width = int(x.max()) + 1
    if height is None:
        height = int(y.max()) + 1

    # Sort events chronologically
    order = np.argsort(t)
    t, x, y, p = t[order], x[order], y[order], p[order]

    t0, t1 = t[0], t[-1]
    total_events = len(t)

    # Determine time bin edges
    if dt is not None:
        edges = np.arange(t0, t1 + dt, dt)
    elif n_frames is not None:
        edges = np.linspace(t0, t1, n_frames + 1)
    elif events_per_frame is not None:
        idx_edges = np.arange(0, total_events + events_per_frame, events_per_frame)
        idx_edges[-1] = total_events
        edges = np.concatenate(([t0], t[idx_edges[1:-1]], [t1]))
    else:
        raise ValueError("Provide exactly one of: dt, n_frames, or events_per_frame.")

    nF = len(edges) - 1
    frames = np.zeros((nF, height, width), dtype=np.float32)

    # Assign events to bins
    bin_ids = np.searchsorted(edges, t, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, nF - 1)

    vals = np.where(p > 0, 1.0, -1.0) if signed else np.ones_like(p, dtype=np.float32)

    # Accumulate pixel values per frame
    for b in range(nF):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        xb, yb, vb = x[mask], y[mask], vals[mask]
        np.add.at(frames[b], (yb, xb), vb)

    # Normalize to [0, 1] for display
    if normalize:
        for i in range(nF):
            f = frames[i]
            m = np.max(np.abs(f)) if signed else f.max()
            if m > 0:
                frames[i] = 0.5 + 0.5 * (f / m) if signed else f / m

    return frames, edges
