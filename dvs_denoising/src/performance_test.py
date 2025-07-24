#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Event Denoising and Visualization Pipeline

This script loads a DVS (Dynamic Vision Sensor) event stream from an HDF5 file,
applies spatiotemporal denoising, converts events into frames, and saves
videos comparing the raw and filtered data.

Modules:
- denoise_events.py: Implements a spatiotemporal density-based denoiser
- convert_events_to_frames.py: Converts events into frames for visualization
- plot.py: Optional event scatter plotting
- imageio: Used for video export
- tqdm: Displays progress bar during filtering and matching

Usage:
    $ python main.py

Author: Xhovani Mali
"""

import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

from accelerated_denoiser import denoise_events
from convert_events_to_frames import events_to_frames
from plot import plot_events


def h5_to_npy(file_path, name):
    with h5py.File(file_path, 'r') as file:
        return file[name][:]


def structured_to_dicts(data):
    return [{'x': int(e['x']), 'y': int(e['y']), 't': float(e['t']), 'p': int(e['p'])} for e in data]


def save_frames_to_video(frames, out_path="output.mp4", fps=30, skip_empty=True, threshold=1e-3):
    """Save a stack of grayscale [0,1] frames to .mp4, skipping empty ones."""
    frames = np.clip(frames, 0, 1)
    frames8 = (frames * 255).astype(np.uint8)

    writer = imageio.get_writer(out_path, fps=fps, codec='libx264')
    skipped = 0
    for i, fr in enumerate(frames8):
        if skip_empty and np.sum(fr) < threshold * fr.size:
            skipped += 1
            continue
        writer.append_data(fr)
    writer.close()
    print(f"Saved video to: {out_path} ({len(frames) - skipped} frames written, {skipped} skipped)")


def main():
    print("Loading data...")
    raw_data = h5_to_npy('../data/9_2.h5', 'events')
    print(f"Loaded {len(raw_data):,} events")
    print(f"Shape: {raw_data.shape}, Dtype: {raw_data.dtype}")
    print("Sample:", raw_data[0])

    events = structured_to_dicts(raw_data)

    print("\nDenoising events...")
    filtered = denoise_events(events, tau_d=10000.0, delta_d=0.05)
    print(f"Original: {len(events):,} | Kept: {len(filtered):,} | Removed: {len(events) - len(filtered):,}")

    print("Extracting removed events...")
    kept_set = set((e['x'], e['y'], e['t']) for e in filtered)
    removed_events = [e for e in tqdm(events, desc="Finding removed", unit="event")
                      if (e['x'], e['y'], e['t']) not in kept_set]

    print("\nGenerating frames...")
    dt = 1000.0
    raw_frames, _ = events_to_frames(events, dt=dt)
    filtered_frames, _ = events_to_frames(filtered, dt=dt)
    print(f"Generated {len(raw_frames)} raw frames and {len(filtered_frames)} filtered frames.")

    PREVIEW_FRAMES = True
    if PREVIEW_FRAMES:
        n_show = min(5, len(filtered_frames))
        for i in range(n_show):
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(raw_frames[i], cmap='gray', origin='lower')
            axs[0].set_title(f"Raw Frame {i}")
            axs[0].axis('off')

            axs[1].imshow(filtered_frames[i], cmap='gray', origin='lower')
            axs[1].set_title(f"Filtered Frame {i}")
            axs[1].axis('off')

            plt.tight_layout()
            plt.show()

    print("Saving videos...")
    save_frames_to_video(raw_frames, "raw_video.mp4", fps=60)
    save_frames_to_video(filtered_frames, "filtered_video.mp4", fps=60)


if __name__ == "__main__":
    main()
