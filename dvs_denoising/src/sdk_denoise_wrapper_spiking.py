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
                # Even lower thresholds and adjusted parameters for reliable propagation
                n_input = self.add_neuron(Vt=3, tm=15, tf=3, Vreset=0, neuron_name=f"input_{x}_{y}")
                n_filter = self.add_neuron(Vt=2, tm=15, tf=3, Vreset=0, neuron_name=f"filter_{x}_{y}")
                n_output = self.add_neuron(Vt=2, tm=15, tf=3, Vreset=0, neuron_name=f"output_{x}_{y}")

                # Much stronger weights for reliable signal transmission
                self.connect_neurons(n_input, n_filter, "ge", weight=15.0, delay=0.1)
                self.connect_neurons(n_filter, n_output, "ge", weight=15.0, delay=0.1)

                self.input_neurons[(x, y)] = n_input
                self.filter_neurons[(x, y)] = n_filter
                self.output_neurons[(x, y)] = n_output

                # Map both integer and string UIDs to coordinates
                self.output_uid_to_xy[n_output.uid] = (x, y)
                # Also map string representation if it's different
                if hasattr(n_output, '__str__'):
                    self.output_uid_to_xy[str(n_output.uid)] = (x, y)

    def get_input_neuron(self, x, y):
        return self.input_neurons.get((x, y), None)

    def output_coord(self, uid):
        """Get coordinates for a neuron UID, handling both int and string UIDs."""
        # Try direct lookup first
        if uid in self.output_uid_to_xy:
            return self.output_uid_to_xy[uid]

        # Try string conversion
        str_uid = str(uid)
        if str_uid in self.output_uid_to_xy:
            return self.output_uid_to_xy[str_uid]

        # Parse the UID string to extract coordinates if it follows the pattern
        if isinstance(uid, str) and '_output_' in uid:
            try:
                # Extract coordinates from string like "(m0,n123)_output_x_y"
                parts = uid.split('_output_')
                if len(parts) == 2:
                    coords = parts[1].split('_')
                    if len(coords) == 2:
                        x, y = int(coords[0]), int(coords[1])
                        if 0 <= x < self.output_width and 0 <= y < self.output_height:
                            return (x, y)
            except (ValueError, IndexError):
                pass

        # If we still can't find it, it might be an input or filter neuron
        # Let's check if it's an output neuron by examining all output neurons
        for (x, y), neuron in self.output_neurons.items():
            if neuron.uid == uid or str(neuron.uid) == str(uid):
                self.output_uid_to_xy[uid] = (x, y)  # Cache it for next time
                return (x, y)

        return (-1, -1)

    def debug_neuron_info(self):
        """Debug function to print neuron UID information."""
        print("\n=== Debug: Neuron UID Information ===")
        print(f"Network has {len(self.output_neurons)} output neurons")
        print("Sample output neuron UIDs:")
        count = 0
        for (x, y), neuron in self.output_neurons.items():
            if count < 5:
                print(f"  Position ({x},{y}): UID = {repr(neuron.uid)} (type: {type(neuron.uid)})")
                count += 1
            else:
                break

        print(f"\nUID mapping dictionary has {len(self.output_uid_to_xy)} entries")
        print("Sample mappings:")
        count = 0
        for uid, coords in self.output_uid_to_xy.items():
            if count < 5:
                print(f"  {repr(uid)} -> {coords}")
                count += 1
            else:
                break
        print("=" * 40)
