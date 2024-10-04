#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
from matplotlib import pyplot as plt
from discrete import *

def traj_diff(traj_1,traj_2):
    return np.sqrt(np.sum((traj_1 - traj_2)**2,axis=1))

def main():
    
    print(sys.getrecursionlimit())
    sys.setrecursionlimit(10000)

    n_run = 100

    frechet_dist_array = np.zeros(n_run)

    fast_frechet = FastDiscreteFrechetMatrix(euclidean)

    for i in range(n_run):
        if i % 20 == 0:
            print(i)
        arr1 = np.loadtxt("data_traj_compare/positions_1_"+str(i)+".csv",delimiter=",")
        arr2 = np.loadtxt("data_traj_compare/positions_2_"+str(i)+".csv",delimiter=",")

        frechet_dist_array[i] = fast_frechet.distance(arr1,arr2)


    np.savetxt("data_traj_compare/frechet_dist_array.csv",frechet_dist_array,delimiter=",")

    # plt.plot(traj_diff(arr1,arr2))
    # plt.show()

    # plt.plot(arr1[:,0],arr1[:,1])
    # plt.plot(arr2[:,0],arr2[:,1])
    # plt.show()

    print("Mean",np.mean(frechet_dist_array))
    print("Std",np.std(frechet_dist_array))

    plt.hist(frechet_dist_array,bins=50)
    plt.title("Frechet distance histogram")
    plt.xlabel("Frechet distance")
    plt.ylabel("Frequency")

    plt.savefig("plots/frechet_dist_hist.pdf")

    plt.show()

    # print(frdist(arr1[:,0:2], arr2[:,0:2]))

def plot():
    #print mean and std of Frechet distance

    frechet_dist_array = np.loadtxt("data_traj_compare/frechet_dist_array.csv",delimiter=",")

    print("Mean",np.mean(frechet_dist_array)/583*100)
    print("Std",np.std(frechet_dist_array)/583*100)
    
    plt.rcParams['font.family'] = 'Palatino'
    plt.hist(frechet_dist_array/583*100,bins=20, color='blue')
    # plt.title("Frechet distance histogram")
    plt.xlabel("Normalized Frechet Distance(%)",fontsize=15)
    plt.ylabel("Frequency",fontsize=15)
    # remove top and right borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plot mean value of Frechet distance
    plt.axvline(x=np.mean(frechet_dist_array)/583*100, color='r', linestyle='dashed', linewidth=1,label="Mean: "+str(round(np.mean(frechet_dist_array)/583*100,2)))
    plt.legend(frameon=False,fontsize=15)
    plt.savefig("plots/frechet_dist_hist.pdf")

    plt.show()

if __name__ == '__main__':
    # main()
    plot()