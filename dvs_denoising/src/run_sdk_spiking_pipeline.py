#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sdk_pipeline.py - Simulates spike-based denoising using Axon SDK with spatial neuron mapping.

Steps:
- Load HDF5 event data
- Infer sensor width/height
- Inject events as spikes into the correct (x,y) neurons
- Run Axon simulation
- Convert spike outputs to frames
- Save video and CSV performance report

Author: Xhovani Mali
"""

import os
import time
import csv
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from axon_sdk.simulator import Simulator
from axon_sdk.primitives.encoders import DataEncoder
from sdk_denoise_wrapper import SpatialSpikingDenoisingNetwork
from convert_events_to_frames import events_to_frames
from main import h5_to_npy


# ---------- Utility Functions ---------- #
def structured_to_dicts(data):
    """Convert structured NumPy array to list of event dictionaries."""
    print("Converting structured array to dicts...")
    return [{'x': int(e['x']), 'y': int(e['y']), 't': float(e['t']), 'p': int(e['p'])}
            for e in tqdm(data, desc="Parsing", unit="event")]


def save_image_plot(frame, out_path):
    """Fallback save as PNG if video is empty."""
    plt.figure(figsize=(6, 6))
    if frame is None:
        plt.text(0.5, 0.5, "No Frames", ha="center", va="center", fontsize=16)
        plt.axis("off")
    else:
        plt.imshow(frame, cmap="gray", interpolation="nearest")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📷 Saved fallback image to: {out_path}")


def save_video(frames, path, fps=30, skip_empty=True, min_intensity_threshold=1e-3):
    """Save frames to MP4 video. If empty, fallback to PNG."""
    if len(frames) == 0:
        print("⚠ No frames to save. Creating fallback PNG.")
        save_image_plot(None, path.replace(".mp4", ".png"))
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = map(int, frames[0].shape)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h), isColor=False)

    if not out.isOpened():
        print(f"Failed to open VideoWriter: {path}")
        save_image_plot(frames[0] if len(frames) > 0 else None, path.replace(".mp4", ".png"))
        return

    saved_count = 0
    for f in tqdm(frames, desc="Writing video"):
        if skip_empty and np.sum(f) < min_intensity_threshold:
            continue
        norm = f.max() if f.max() > 0 else 1
        f_uint8 = (255 * f / norm).astype(np.uint8)
        out.write(f_uint8)
        saved_count += 1

    out.release()
    print(f"Video saved: {path} ({saved_count} frames written)")


def save_csv(report, path):
    """Save performance report to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(report.keys())
        writer.writerow(report.values())
    print(f"CSV saved: {path}")


# ---------- Main Simulation ---------- #
if __name__ == "__main__":
    report = {}
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(base_dir, "..", "data"))

    # === Load Data === #
    print("Loading HDF5...")
    raw = h5_to_npy(os.path.join(data_dir, "9_2.h5"), "events")
    events = structured_to_dicts(raw)
    report["events_loaded"] = len(events)

    # === Crop Resolution for Speed === #
    crop_w, crop_h = 64, 64
    events = [e for e in events if e["x"] < crop_w and e["y"] < crop_h]
    events = events[:20000]  # Limit to 20K events for fast sim
    report["events_used"] = len(events)
    print(f"Using {len(events):,} events within {crop_w}x{crop_h} region")

    # === Initialize Network === #
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = SpatialSpikingDenoisingNetwork(encoder, width=crop_w, height=crop_h)
    sim = Simulator(net, encoder, dt=0.001)

    # === Inject Events === #
    print("⚡ Encoding & injecting spikes...")
    t0 = time.time()
    for e in tqdm(events, desc="Injecting events", unit="ev"):
        neuron = net.get_input_neuron(e["x"], e["y"])
        if neuron:
            spikes = encoder.encode_value(0.5)
            for s in spikes:
                sim.apply_input_spike(neuron=neuron, t=e["t"] * 1e-6 + s)
    t1 = time.time()
    report["inject_time_sec"] = round(t1 - t0, 4)

    # === Run Simulation (max 2s) === #
    print("Simulating network...")
    max_sim_time = min(events[-1]["t"] * 1e-6 + 0.1, 2.0)
    t2 = time.time()
    sim.simulate(simulation_time=max_sim_time)
    t3 = time.time()
    report["simulation_time_sec"] = round(t3 - t2, 4)

    # === Extract Output Spikes === #
    print("Extracting output spikes...")
    out_events = []
    for uid, spikes in sim.spike_log.items():
        x, y = net.output_coord(uid)
        for t in spikes:
            out_events.append({"x": x, "y": y, "t": t * 1e6, "p": 1})
    report["output_events"] = len(out_events)

    # === Convert to Frames === #
    frames, _ = events_to_frames(out_events, dt=1000.0)
    report["n_frames"] = len(frames)

    # === Save Output === #
    save_video(frames, path=os.path.join(data_dir, "axon_spatial_output.mp4"), fps=30)
    save_csv(report, path=os.path.join(data_dir, "axon_spatial_report.csv"))

    print("\n=== Simulation Done ===")
    for k, v in report.items():
        print(f"{k:24}: {v}")
