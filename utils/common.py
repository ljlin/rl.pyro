import pickle
import numpy as np
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
    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    plt.plot(x, mean, color=color, label=label)
    plt.fill_between(x, mean-std, mean+std, color=color, alpha=0.3)

# Update a target network using a source network
def update(target, source):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(p.data)
