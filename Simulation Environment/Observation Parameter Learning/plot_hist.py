import numpy as np
# import pygame
# from pygame.locals import *
# from robot import *
import random
from matplotlib import pyplot as plt
import scipy.stats as stats

def main():
    traj = np.loadtxt("data_2/demo_positions_0_09_"+str(0)+".csv", delimiter=",")

    thetas = traj[:,3] - traj[:,2]
    x = np.linspace(0 - 5*0.9, 0 + 5*0.9, 100)

    plt.hist(thetas,bins=10,density=True,label="Histogram")
    plt.plot(x, stats.norm.pdf(x, 0, 0.9),label="N(0,0.9)")
    plt.legend()
    plt.show()

if(__name__ == '__main__'):
    main()
