import pickle
import numpy
import matplotlib.pyplot as plt
def save(var, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
        return b

def safe_filename(filename):
    keepcharacters = ('.','_','-',"=")
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip("_")

# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(x, y, color, label):
    mean = numpy.mean(y, axis=0)
    std = numpy.std(y, axis=0)
    plt.plot(x, mean, color=color, label=label)
    plt.fill_between(x, mean-std, mean+std, color=color, alpha=0.3)

# Update a target network using a source network
def update(target, source):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(p.data)

def train_and_plot(train, SEEDS, filename, info, MODE, x, show = False):
    curves = [train(seed) for seed in SEEDS]
    with open(f'{filename}.csv', 'w') as csv:
        numpy.savetxt(csv, numpy.asarray(curves), delimiter=',')
    # Plot the curve for the given seeds
    plt.figure(dpi=120)
    label = f"{MODE}{'-' + info if info else ''}"
    plot_arrays(x, curves, 'b', label)
    plt.legend(loc='best')
    plt.savefig(f'{filename}.png')
    if show:
        plt.show()