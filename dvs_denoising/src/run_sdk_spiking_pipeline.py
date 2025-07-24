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
    print(f"Saved fallback image to: {out_path}")


def save_video(frames, path, fps=30, skip_empty=True, min_intensity_threshold=1e-3):
    """Save frames to MP4 video. If empty, fallback to PNG."""
    if len(frames) == 0:
        print("Warning: No frames to save. Creating fallback PNG.")
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

    # === Determine Full Resolution === #
    print("Analyzing event data...")
    max_x = max(e['x'] for e in events[:10000])  # Sample first 10K events for speed
    max_y = max(e['y'] for e in events[:10000])
    print(f"Detected sensor resolution: {max_x + 1}x{max_y + 1}")

    # Use full resolution but limit to reasonable size for memory
    full_w, full_h = min(max_x + 1, 640), min(max_y + 1, 480)  # Cap at 640x480
    events = [e for e in events if e["x"] < full_w and e["y"] < full_h]

    # Process all events (remove the 20K limit)
    print(f"Processing all {len(events):,} events within {full_w}x{full_h} region")
    report["events_used"] = len(events)
    report["resolution"] = f"{full_w}x{full_h}"

    # === Initialize Network === #
    encoder = DataEncoder(Tmin=5.0, Tcod=50.0)  # Stronger encoding
    net = SpatialSpikingDenoisingNetwork(encoder, width=full_w, height=full_h)
    sim = Simulator(net, encoder, dt=0.001)

    # Debug network setup
    print(f"Network initialized with {full_w * full_h:,} neurons per layer")
    print(f"Total neurons: {full_w * full_h * 3:,} (input + filter + output)")
    net.debug_neuron_info()

    # === Inject Events === #
    print("⚡ Encoding & injecting spikes...")
    print("Warning: Processing full dataset - this may take several minutes...")
    t0 = time.time()
    spike_count = 0

    # Process events in batches for memory efficiency
    batch_size = 10000
    for i in range(0, len(events), batch_size):
        batch = events[i:i + batch_size]
        for e in tqdm(batch, desc=f"Batch {i // batch_size + 1}/{(len(events) - 1) // batch_size + 1}", unit="ev"):
            neuron = net.get_input_neuron(e["x"], e["y"])
            if neuron:
                # Use multiple strong spikes per event for reliable activation
                base_time = e["t"] * 1e-6

                # Inject multiple spikes in a burst for stronger stimulation
                for j in range(3):  # 3 spikes per event
                    spike_time = base_time + j * 0.0005  # 0.5ms apart
                    sim.apply_input_spike(neuron=neuron, t=spike_time)
                    spike_count += 1

    print(f"Injected {spike_count:,} total spikes for {len(events):,} events")
    print(f"Average: {spike_count / len(events):.1f} spikes per event")
    t1 = time.time()
    report["inject_time_sec"] = round(t1 - t0, 4)
    report["total_spikes_injected"] = spike_count

    # === Run Simulation (extended time for propagation) === #
    print("Simulating network...")
    max_sim_time = min(events[-1]["t"] * 1e-6 + 0.5, 5.0)  # Extended simulation time
    print(f"Simulation time: {max_sim_time:.2f} seconds")
    print("This may take 10-25 minutes for full dataset. Progress updates every 30 seconds...")

    t2 = time.time()

    # Run simulation with progress monitoring
    import threading
    import sys


    def progress_monitor():
        """Monitor simulation progress and provide updates"""
        start_time = time.time()
        update_interval = 30  # seconds

        while True:
            time.sleep(update_interval)
            elapsed = time.time() - start_time

            # Check if simulation is still running by looking at thread count
            if threading.active_count() <= 2:  # Main thread + this monitor
                break

            print(f"Simulation running... {elapsed / 60:.1f} minutes elapsed")

            # Estimate progress (very rough based on time)
            if elapsed < 600:  # First 10 minutes
                progress = min(20, elapsed / 30)
                print(f"   Estimated progress: ~{progress:.0f}%")
            elif elapsed < 1200:  # 10-20 minutes
                progress = 20 + min(60, (elapsed - 600) / 10)
                print(f"   Estimated progress: ~{progress:.0f}%")
            else:  # > 20 minutes
                print(f"   Progress: Final stages (spike propagation)")

            sys.stdout.flush()


    # Start progress monitor in background
    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()

    print("Starting simulation...")

    # Check if we can get any immediate feedback
    try:
        # Try to get some early simulation info if available
        print(f"Processing {spike_count:,} spikes through {full_w * full_h * 3:,} neurons...")
        print(f"Expected computation: ~{(spike_count * full_w * full_h / 1e9):.1f}B operations")

        # Start the actual simulation
        sim.simulate(simulation_time=max_sim_time)

    except Exception as e:
        print(f"Simulation error: {e}")
        raise

    t3 = time.time()
    simulation_minutes = (t3 - t2) / 60
    print(f"Simulation completed in {simulation_minutes:.1f} minutes!")
    report["simulation_time_sec"] = round(t3 - t2, 4)

    # === Extract Output Spikes === #
    print("Extracting output spikes...")
    out_events = []
    unknown_uids = set()
    valid_uids = set()
    input_uids = set()
    filter_uids = set()
    output_uids = set()

    print(f"Total spike log entries: {len(sim.spike_log)}")

    # Count total spikes by neuron type
    input_spike_count = 0
    filter_spike_count = 0
    output_spike_count = 0

    # Categorize all UIDs in spike log
    for uid, spikes in sim.spike_log.items():
        uid_str = str(uid)
        spike_count = len(spikes)

        if '_input_' in uid_str:
            input_uids.add(uid)
            input_spike_count += spike_count
        elif '_filter_' in uid_str:
            filter_uids.add(uid)
            filter_spike_count += spike_count
        elif '_output_' in uid_str:
            output_uids.add(uid)
            output_spike_count += spike_count
        else:
            unknown_uids.add(uid)

    print(f"Spike breakdown:")
    print(f"  Input neurons: {len(input_uids)} neurons, {input_spike_count} total spikes")
    print(f"  Filter neurons: {len(filter_uids)} neurons, {filter_spike_count} total spikes")
    print(f"  Output neurons: {len(output_uids)} neurons, {output_spike_count} total spikes")
    print(f"  Unknown neurons: {len(unknown_uids)}")

    # Show some examples of neurons that fired
    if input_spike_count > 0:
        active_inputs = [(uid, len(sim.spike_log[uid])) for uid in input_uids if len(sim.spike_log[uid]) > 0]
        print(f"Sample active input neurons: {active_inputs[:3]}")

    if filter_spike_count > 0:
        active_filters = [(uid, len(sim.spike_log[uid])) for uid in filter_uids if len(sim.spike_log[uid]) > 0]
        print(f"Sample active filter neurons: {active_filters[:3]}")

    if output_spike_count > 0:
        active_outputs = [(uid, len(sim.spike_log[uid])) for uid in output_uids if len(sim.spike_log[uid]) > 0]
        print(f"Sample active output neurons: {active_outputs[:3]}")

    # Extract events only from output neurons
    for uid in output_uids:
        spikes = sim.spike_log[uid]
        if len(spikes) == 0:
            continue

        x, y = net.output_coord(uid)
        if x == -1 and y == -1:
            print(f"Warning: Output neuron {uid} has invalid coordinates!")
            continue

        valid_uids.add(uid)
        for t in spikes:
            out_events.append({"x": x, "y": y, "t": t * 1e6, "p": 1})

    print(f"Valid output neuron UIDs found: {len(valid_uids)}")

    if len(output_uids) == 0 or output_spike_count == 0:
        print("⚠ No output neurons fired during simulation!")
        print("This might indicate:")
        print("  - Thresholds too high")
        print("  - Insufficient input stimulation")
        print("  - Network connectivity issues")

        if input_spike_count == 0:
            print("  - No input spikes detected - check spike injection!")
        elif filter_spike_count == 0:
            print("  - Input spikes present but not reaching filter neurons")
        else:
            print("  - Filter neurons active but not reaching output neurons")

    report["output_events"] = len(out_events)
    report["spike_counts"] = {
        "input": input_spike_count,
        "filter": filter_spike_count,
        "output": output_spike_count
    }
    print(f"Extracted {len(out_events)} valid output events")

    # === Convert to Frames === #
    if len(out_events) > 0:
        frames, _ = events_to_frames(out_events, dt=1000.0)
        report["n_frames"] = len(frames)
    else:
        print("Warning: No valid output events to convert to frames")
        frames = []
        report["n_frames"] = 0

    # === Save Output === #
    save_video(frames, path=os.path.join(data_dir, "axon_spatial_output_9_2_full.mp4"), fps=30)
    save_csv(report, path=os.path.join(data_dir, "axon_spatial_report_9_2_full.csv"))

    print("\n=== Simulation Done ===")
    for k, v in report.items():
        print(f"{k:24}: {v}")
