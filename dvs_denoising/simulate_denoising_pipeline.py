# simulate_denoising_pipeline.py
from axon_sdk.simulator import Simulator
from axon_sdk.primitives.encoders import DataEncoder
from axon_sdk_denoise_module import DenoisingPreprocessor
from main_performance import h5_to_npy, structured_to_dicts
import matplotlib.pyplot as plt
from time import perf_counter


encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
module = DenoisingPreprocessor(encoder)
sim = Simulator(module, encoder)

raw_data = h5_to_npy('./data/9_2.h5', 'events')
events = structured_to_dicts(raw_data)

frames = module.process(events)

print(f"✅ Processed {len(events):,} events -> {len(frames)} frames")

plt.imshow(frames[0], cmap='gray')
plt.title("Denoised Frame 0")
plt.colorbar()
plt.show()

t0 = perf_counter()
frames = module.process(events)
t1 = perf_counter()
print(f"⏱ Axon pipeline time: {t1 - t0:.2f}s")
