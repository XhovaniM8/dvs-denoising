#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Lightweight Event Denoising Pipeline

Loads a DVS (Dynamic Vision Sensor) event stream from HDF5, applies a
spatiotemporal denoising filter, converts events to grayscale frames,
and writes raw and denoised videos to disk.

Author: Xhovani Mali
"""

import h5py
import numpy as np
import imageio

from accelerated_denoiser import denoise_events
from convert_events_to_frames import events_to_frames

def h5_to_npy(file_path, name):
    with h5py.File(file_path, 'r') as file:
        return file[name][:]

def structured_to_dicts(data):
    return [{'x': int(e['x']), 'y': int(e['y']), 't': float(e['t']), 'p': int(e['p'])} for e in data]

def save_frames_to_video(frames, out_path="output.mp4", fps=30, skip_empty=True, threshold=1e-3):
    """Save grayscale frames [0,1] to .mp4, skipping nearly empty frames if enabled."""
    frames = np.clip(frames, 0, 1)
    frames8 = (frames * 255).astype(np.uint8)

    writer = imageio.get_writer(out_path, fps=fps, codec='libx264')
    skipped = 0
    for fr in frames8:
        if skip_empty and np.sum(fr) < threshold * fr.size:
            skipped += 1
            continue
        writer.append_data(fr)
    writer.close()
    print(f"Saved video to: {out_path} ({len(frames) - skipped} written, {skipped} skipped)")

def main():
    print("Loading data...")
    raw_data = h5_to_npy('../data/9_2.h5', 'events')
    print(f"Loaded {len(raw_data):,} events")

    events = structured_to_dicts(raw_data)

    print("Running denoiser...")
    filtered = denoise_events(events, tau_d=10000.0, delta_d=0.05)
    print(f"Original: {len(events):,} | Kept: {len(filtered):,} | Removed: {len(events) - len(filtered):,}")

    print("Converting to frames...")
    dt = 1000.0
    raw_frames, _ = events_to_frames(events, dt=dt)
    filtered_frames, _ = events_to_frames(filtered, dt=dt)
    print(f"Raw frames: {len(raw_frames)} | Filtered frames: {len(filtered_frames)}")

    print("Saving videos...")
    save_frames_to_video(raw_frames, "../data/raw_video.mp4", fps=60)
    save_frames_to_video(filtered_frames, "../data/filtered_video.mp4", fps=60)

    print("Done.")

if __name__ == "__main__":
    main()
