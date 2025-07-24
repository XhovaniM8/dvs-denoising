#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_performance.py - Benchmarking for DVS Denoising

Measures execution time and basic stats for each stage of the DVS denoising pipeline:
- Data loading
- Denoising
- Frame generation

Outputs a CSV report of timings and event counts to ./data/performance_report.csv

Author: Xhovani Mali
"""

import h5py
import numpy as np
import time
import csv
from tqdm import tqdm
from denoise_events import denoise_events
from event_to_frames import events_to_frames


def h5_to_npy(file_path, name):
    """Load dataset from HDF5 file into NumPy structured array."""
    with h5py.File(file_path, 'r') as file:
        return file[name][:]


def structured_to_dicts(data):
    """Convert structured NumPy array to list of event dictionaries."""
    print("Converting structured array to dicts...")
    result = []
    for e in tqdm(data, desc="Parsing events", unit="event"):
        result.append({'x': int(e['x']), 'y': int(e['y']), 't': float(e['t']), 'p': int(e['p'])})
    return result


def benchmark():
    report = {}

    # === Load Data ===
    print("\nLoading data...")
    t0 = time.time()
    raw_data = h5_to_npy('./data/9_2.h5', 'events')
    t1 = time.time()
    print(f"Loaded {len(raw_data):,} events.")
    events = structured_to_dicts(raw_data)
    t2 = time.time()
    report["events_loaded"] = len(events)
    report["load_time_sec"] = round(t1 - t0, 4)
    report["parse_time_sec"] = round(t2 - t1, 4)

    # === Denoising ===
    print("\nRunning denoising...")
    t3 = time.time()
    filtered = denoise_events(events, tau_d=10000.0, delta_d=0.05)
    t4 = time.time()
    print(f"Kept {len(filtered):,} | Removed {len(events) - len(filtered):,}")
    report["denoise_time_sec"] = round(t4 - t3, 4)
    report["events_kept"] = len(filtered)
    report["events_removed"] = len(events) - len(filtered)

    # === Frame Generation ===
    print("\nGenerating frames...")
    t5 = time.time()
    raw_frames, _ = events_to_frames(events, dt=1000.0)
    filtered_frames, _ = events_to_frames(filtered, dt=1000.0)
    t6 = time.time()
    print(f"Generated {len(raw_frames)} frames.")
    report["frame_gen_time_sec"] = round(t6 - t5, 4)
    report["n_frames"] = len(raw_frames)

    return report


def save_report(report, filename="./data/performance_report.csv"):
    """Write performance report as CSV to the data folder."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(report.keys())
        writer.writerow(report.values())
    print(f"\n✅ Saved performance report to: {filename}")


if __name__ == "__main__":
    print("=== DVS Denoising Benchmark ===")
    results = benchmark()
    print("\n=== Results ===")
    for key, value in results.items():
        print(f"{key:25s}: {value}")
    save_report(results)
