#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Lightweight Event Denoising Pipeline

Loads a DVS (Dynamic Vision Sensor) event stream from HDF5, applies a
spatiotemporal denoising filter, and converts events to grayscale frames.

Intended for fast benchmarking or non-visual inspection.

Author: Xhovani Mali
"""

import h5py
from accelerated_denoiser import denoise_events
from convert_events_to_frames import events_to_frames

def h5_to_npy(file_path, name):
    with h5py.File(file_path, 'r') as file:
        return file[name][:]

def structured_to_dicts(data):
    return [{'x': int(e['x']), 'y': int(e['y']), 't': float(e['t']), 'p': int(e['p'])} for e in data]

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

    print("Done.")

if __name__ == "__main__":
    main()
