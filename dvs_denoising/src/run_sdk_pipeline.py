#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
axon_pipeline_performance.py - Benchmarks Axon SDK pipeline w/ custom denoiser

Steps:
- Load HDF5 event data
- Convert to list-of-dicts format
- Run DenoisingPreprocessor from axos_sdk_denoise_module
- Convert output to grayscale frames
- Save frames to video
- Output CSV performance report

Author: Xhovani Mali
"""

import time
import csv
import cv2
import numpy as np
from tqdm import tqdm
from axon_sdk.simulator import Simulator
from axon_sdk.primitives.encoders import DataEncoder
from sdk_denoise_wrapper import DenoisingPreprocessor
from main import h5_to_npy

# === Helpers ===
def structured_to_dicts(data):
    """Convert structured NumPy array to list of event dictionaries."""
    print("Converting structured array to dicts...")
    return [{'x': int(e['x']), 'y': int(e['y']), 't': float(e['t']), 'p': int(e['p'])} for e in tqdm(data, desc="Parsing", unit="event")]

def save_video(frames, path="data/output.mp4", fps=30, skip_empty=True, min_intensity_threshold=1e-3):
    """
    Save grayscale frames as black-and-white video.

    Args:
        frames: list or np.ndarray of 2D arrays (grayscale)
        path: output .mp4 path
        fps: frames per second
        skip_empty: whether to skip empty (all-zero) or near-zero frames
        min_intensity_threshold: minimum total intensity to consider a frame non-empty
    """
    if len(frames) == 0:
        print("No frames to save.")
        return

    h, w = frames[0].shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), isColor=False)

    saved_count = 0
    for f in tqdm(frames, desc="Writing video"):
        if skip_empty and np.sum(f) < min_intensity_threshold:
            continue  # Skip black frames

        f_uint8 = (255 * f / (f.max() if f.max() > 0 else 1)).astype(np.uint8)
        out.write(f_uint8)
        saved_count += 1

    out.release()
    print(f"Video saved: {path} ({saved_count} frames written)")


def save_csv(report, path="data/axon_performance_report.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(report.keys())
        writer.writerow(report.values())
    print(f"CSV saved: {path}")

# === Main pipeline ===
if __name__ == "__main__":
    report = {}

    # --- Load and parse events ---
    print("\nLoading HDF5...")
    t0 = time.time()
    raw = h5_to_npy('../data/9_2.h5', 'events')
    t1 = time.time()
    events = structured_to_dicts(raw)
    t2 = time.time()

    report['events_loaded'] = len(events)
    report['load_time_sec'] = round(t1 - t0, 3)
    report['parse_time_sec'] = round(t2 - t1, 3)

    # --- Setup Axon pipeline ---
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    module = DenoisingPreprocessor(encoder)
    sim = Simulator(module, encoder)

    # --- Denoising + encoding ---
    print("\nRunning Axon pipeline...")
    t3 = time.time()
    frames = module.process(events)
    t4 = time.time()

    report['n_frames'] = len(frames)
    report['axon_time_sec'] = round(t4 - t3, 3)
    report.update(module.stats)  # if your module provides extra stats

    # --- Save video + CSV ---
    save_video(frames, path="../data/output.mp4", fps=30)
    save_csv(report, path="../data/axon_performance_report.csv")

    print("\n=== Done ===")
    for k, v in report.items():
        print(f"{k:20}: {v}")
