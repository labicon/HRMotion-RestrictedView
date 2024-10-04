import numpy as np
import matplotlib.pyplot as plt

def plot_data(trajectory):
    plt.plot(trajectory[:,0], trajectory[:,1], 'r')
    plt.show()

def plot_controls(controls):
    plt.plot(controls[:,0], 'r')
    plt.plot(controls[:,1], 'b')
    plt.show()

def main():
    trajectory = np.loadtxt("data/positions.csv", delimiter=",", dtype=float)
    controls = np.loadtxt("data/controls.csv", delimiter=",", dtype=float)

    plot_controls(controls)

if(__name__ == '__main__'):
    main()