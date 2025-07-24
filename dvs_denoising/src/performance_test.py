#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
performance_test.py - Benchmarks DVS denoising pipeline

Measures:
- Load + parse time
- Denoising time
- Frame generation time
- Event statistics

Outputs:
- CSV report to `performance_report.csv`

Author: Xhovani Mali
"""

import h5py
import time
import csv
from accelerated_denoiser import denoise_events
from convert_events_to_frames import events_to_frames


def h5_to_npy(file_path, name):
    with h5py.File(file_path, 'r') as file:
        return file[name][:]


def structured_to_dicts(data):
    return [{'x': int(e['x']), 'y': int(e['y']), 't': float(e['t']), 'p': int(e['p'])} for e in data]


def save_csv(report, path="../data/performance_report.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(report.keys())
        writer.writerow(report.values())
    print(f"Saved performance report to {path}")


def main():
    report = {}

    # --- Load and parse ---
    print("Loading HDF5...")
    t0 = time.time()
    raw_data = h5_to_npy('../data/9_2.h5', 'events')
    t1 = time.time()
    events = structured_to_dicts(raw_data)
    t2 = time.time()

    report['events_loaded'] = len(events)
    report['load_time_sec'] = round(t1 - t0, 4)
    report['parse_time_sec'] = round(t2 - t1, 4)

    # --- Denoise ---
    print("Running denoiser...")
    t3 = time.time()
    filtered = denoise_events(events, tau_d=10000.0, delta_d=0.05)
    t4 = time.time()
    report['denoise_time_sec'] = round(t4 - t3, 4)
    report['events_kept'] = len(filtered)
    report['events_removed'] = len(events) - len(filtered)

    # --- Frame generation ---
    print("Generating frames...")
    dt = 1000.0
    t5 = time.time()
    filtered_frames, _ = events_to_frames(filtered, dt=dt)
    t6 = time.time()
    report['frame_gen_time_sec'] = round(t6 - t5, 4)
    report['n_frames'] = len(filtered_frames)

    # --- Output ---
    save_csv(report)

    print("\n=== Benchmark Complete ===")
    for k, v in report.items():
        print(f"{k:22}: {v}")


if __name__ == "__main__":
    main()
