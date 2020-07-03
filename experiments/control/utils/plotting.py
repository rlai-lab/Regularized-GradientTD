def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)

def plot(ax, data, label=None, color=None):
    mean, ste, runs = data
    base, = ax.plot(mean, label=label, color=color, linewidth=2)
    (low_ci, high_ci) = confidenceInterval(mean, ste)
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color=base.get_color(), alpha=0.4)
