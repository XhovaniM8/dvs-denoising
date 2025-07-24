# sdk_denoise_wrapper.py

from axon_sdk.primitives.networks import SpikingNetworkModule
from accelerated_denoiser import denoise_events
from convert_events_to_frames import events_to_frames
import time

class DenoisingPreprocessor(SpikingNetworkModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder  # required by Axon
        self.stats = {}  # populated during processing

    def process(self, raw_events):
        print("[Axon] Preprocessing events...")

        # --- Denoising ---
        t0 = time.time()
        filtered = denoise_events(raw_events)
        t1 = time.time()

        # --- Frame Generation ---
        frames, _ = events_to_frames(filtered, dt=1000.0)
        t2 = time.time()

        # --- Stats ---
        self.stats['events_kept'] = len(filtered)
        self.stats['events_removed'] = len(raw_events) - len(filtered)
        self.stats['denoise_time_sec'] = round(t1 - t0, 4)
        self.stats['frame_gen_time_sec'] = round(t2 - t1, 4)

        return frames
