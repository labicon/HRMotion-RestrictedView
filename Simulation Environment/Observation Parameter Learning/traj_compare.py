import numpy as np
# import pygame
# from pygame.locals import *
# from robot import *
import random
from matplotlib import pyplot as plt

def traj_diff(traj_1,traj_2):
    return np.sqrt(np.sum((traj_1 - traj_2)**2,axis=1))/583

def read_traj(l,n):
    traj_1 = np.loadtxt("data/orig_traj/positions_param_"+str(l)+"_"+str(n)+".csv", delimiter=",")
    traj_2 = np.loadtxt("data/pred_traj/positions_param_"+str(l)+"_"+str(n)+".csv", delimiter=",")

    l1 = traj_1.shape[0]
    l2 = traj_2.shape[0]

    #print(l1,l2)

    if l1<=l2:
        traj_2 = traj_2[0:l1,0:2]
        traj_1 = traj_1[:,0:2]
    else:
        traj_1 = traj_1[0:l2,0:2]
        traj_2 = traj_2[:,0:2]

    # print(traj_1.shape)
    # print(traj_2.shape)

    return [traj_1,traj_2]

def main():

    distance_arrays = []

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    for l in range(4):
        ax = axs[l // 2, l % 2]  # Get the current subplot
        for i in range(200):
            [traj_1, traj_2] = read_traj(l,i)
            array = traj_diff(traj_1,traj_2)
            #distance_arrays.append(array)
            # plt.plot(traj_1[:,0],traj_1[:,1],color='blue')
            # plt.plot(traj_2[:,0],traj_2[:,1],color='orange')
            ax.plot(array, color="blue")
        ax.set_title(f'Parameter {l+1}')
        ax.set_xlabel('Time')
        ax.set_ylim(0,1.3)
        ax.set_ylabel('Difference')

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the plot
    plt.savefig("traj_compare.pdf")

if(__name__ == '__main__'):
    main()
