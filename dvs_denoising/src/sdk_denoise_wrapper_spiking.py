from axon_sdk.primitives.networks import SpikingNetworkModule
from axon_sdk.primitives.elements import ExplicitNeuron

class SpatialSpikingDenoisingNetwork(SpikingNetworkModule):
    def __init__(self, encoder, width, height):
        super().__init__("SpatialSpikingDenoiser")
        self.encoder = encoder
        self.output_width = width
        self.output_height = height
        self.input_neurons = {}
        self.filter_neurons = {}
        self.output_neurons = {}
        self.output_uid_to_xy = {}

        for y in range(height):
            for x in range(width):
                n_input = self.add_neuron(Vt=10, tm=20, tf=5, Vreset=0, neuron_name=f"input_{x}_{y}")
                n_filter = self.add_neuron(Vt=10, tm=20, tf=5, Vreset=0, neuron_name=f"filter_{x}_{y}")
                n_output = self.add_neuron(Vt=10, tm=20, tf=5, Vreset=0, neuron_name=f"output_{x}_{y}")

                self.connect_neurons(n_input, n_filter, "ge", weight=5.0, delay=1.0)
                self.connect_neurons(n_filter, n_output, "ge", weight=10.0, delay=1.0)

                self.input_neurons[(x, y)] = n_input
                self.filter_neurons[(x, y)] = n_filter
                self.output_neurons[(x, y)] = n_output
                self.output_uid_to_xy[n_output.uid] = (x, y)

    def get_input_neuron(self, x, y):
        return self.input_neurons.get((x, y), None)

    def output_coord(self, uid: int):
        return self.output_uid_to_xy.get(uid, (-1, -1))
