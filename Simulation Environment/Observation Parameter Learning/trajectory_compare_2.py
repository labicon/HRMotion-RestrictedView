import numpy as np
# import pygame
# from pygame.locals import *
# from robot import *
import random
from matplotlib import pyplot as plt

def traj_diff(traj_1,traj_2):
    return np.sqrt(np.sum((traj_1 - traj_2)**2,axis=1))/583

def read_traj(n):
    traj_1 = np.loadtxt("data_2/positions_0_09_"+str(n)+".csv", delimiter=",")
    traj_2 = np.loadtxt("data_2/positions_pred_0_09_"+str(n)+".csv", delimiter=",")

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

    for i in range(200):
        [traj_1, traj_2] = read_traj(i)
        array = traj_diff(traj_1,traj_2)
        #distance_arrays.append(array)
        # plt.plot(traj_1[:,0],traj_1[:,1],color='blue')
        # plt.plot(traj_2[:,0],traj_2[:,1],color='orange')
        plt.plot(array, color = "blue")

    plt.show()

if(__name__ == '__main__'):
    main()
