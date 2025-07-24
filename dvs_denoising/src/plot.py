import matplotlib.pyplot as plt

def plot_events(events, title="Events", color='blue', show=True):
    xs = [e['x'] for e in events]
    ys = [e['y'] for e in events]
    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys, s=10, c=color, alpha=0.6)
    plt.title(title)
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.gca().invert_yaxis()
    if show:
        plt.show()

def plot_combined(events, filtered_events):
    kept_set = set((e['x'], e['y'], e['t']) for e in filtered_events)
    removed = [e for e in events if (e['x'], e['y'], e['t']) not in kept_set]

    plt.figure(figsize=(6, 6))
    plt.scatter([e['x'] for e in removed], [e['y'] for e in removed], s=10, c='red', label='Removed', alpha=0.5)
    plt.scatter([e['x'] for e in filtered_events], [e['y'] for e in filtered_events], s=10, c='green', label='Kept', alpha=0.5)
    plt.legend()
    plt.title("Denoising Result")
    plt.gca().invert_yaxis()
    plt.show()

