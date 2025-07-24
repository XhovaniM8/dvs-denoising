# sdk_denoise_wrapper.py
from axon_sdk.primitives.networks import SpikingNetworkModule
from accelerated_denoiser import denoise_events
from convert_events_to_frames import events_to_frames

class DenoisingPreprocessor(SpikingNetworkModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.stats = {}

    def process(self, raw_events):
        print("[Axon] Preprocessing events...")
        n_raw = len(raw_events)

        filtered = denoise_events(raw_events)
        n_kept = len(filtered)
        self.stats['events_kept'] = n_kept
        self.stats['events_removed'] = n_raw - n_kept
        self.stats['percent_retained'] = round(100 * n_kept / n_raw, 2)

        frames, edges = events_to_frames(filtered, dt=1000.0)
        self.stats['frame_count'] = len(frames)
        self.stats['frame_duration_us'] = round(edges[1] - edges[0], 2) if len(edges) > 1 else None

        return frames
