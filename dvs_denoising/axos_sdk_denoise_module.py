# axon_sdk_denoise_module.py
from axon.axon_sdk.base import SpikingNetworkModule

class DenoisingPreprocessor(SpikingNetworkModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder  # not used, but required by base class

    def process(self, raw_events):
        from denoise_events import denoise_events
        from event_to_frames import events_to_frames

        print("[Axon] Preprocessing events...")
        filtered = denoise_events(raw_events)
        frames, _ = events_to_frames(filtered, dt=1000.0)
        return frames
