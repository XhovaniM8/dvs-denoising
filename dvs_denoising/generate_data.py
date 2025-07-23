import random

def generate_synthetic_events(num_valid=100, num_noise=50):
    # Generate valid events clustered near (40, 40)
    valid_events = [{
        'x': int(np.random.normal(40, 2)),
        'y': int(np.random.normal(40, 2)),
        't': random.uniform(0, 1),
        'p': 1
    } for _ in range(num_valid)]

    # Generate noise events scattered randomly
    noise_events = [{
        'x': random.randint(0, 80),
        'y': random.randint(0, 80),
        't': random.uniform(0, 1),
        'p': 1
    } for _ in range(num_noise)]

    return sorted(valid_events + noise_events, key=lambda e: e['t'])
