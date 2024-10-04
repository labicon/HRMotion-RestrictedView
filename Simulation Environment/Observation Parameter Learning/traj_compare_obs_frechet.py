import numpy as np
# import pygame
# from pygame.locals import *
# from robot import *
# import random
from matplotlib import pyplot as plt
from discrete import *

def traj_diff(traj_1,traj_2):
    return np.sqrt(np.sum((traj_1 - traj_2)**2,axis=1))/583

def read_traj(l,n):
    traj_1 = np.loadtxt("data/orig_traj/positions_param_"+str(l)+"_"+str(n)+".csv", delimiter=",")
    traj_2 = np.loadtxt("data/pred_traj/positions_param_"+str(l)+"_"+str(n)+".csv", delimiter=",")

    # l1 = traj_1.shape[0]
    # l2 = traj_2.shape[0]

    # #print(l1,l2)

    # if l1<=l2:
    #     traj_2 = traj_2[0:l1,0:2]
    #     traj_1 = traj_1[:,0:2]
    # else:
    #     traj_1 = traj_1[0:l2,0:2]
    #     traj_2 = traj_2[:,0:2]

    # print(traj_1.shape)
    # print(traj_2.shape)

    return [traj_1,traj_2]

def main():

    distance_arrays = np.zeros((4,200))

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    fast_frechet = FastDiscreteFrechetMatrix(euclidean)

    for l in range(4):
        ax = axs[l // 2, l % 2]  # Get the current subplot
        for i in range(200):
            if i % 10 == 0:
                print(f"Parameter {l+1} - Trajectory {i + 1}")
            [traj_1, traj_2] = read_traj(l,i)
            distance_arrays[l,i] = fast_frechet.distance(traj_1,traj_2)/583
        
        plt.hist(distance_arrays[l,:], bins=20, color="blue")
        ax.set_xlabel('Normalized Frechet Distance wrt total trajectory length')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Parameter {l+1}')

    np.savetxt("data/frechet_distance.csv", distance_arrays, delimiter=",")
    
    # plt.show()

            # array = traj_diff(traj_1,traj_2)
            #distance_arrays.append(array)
            # plt.plot(traj_1[:,0],traj_1[:,1],color='blue')
            # plt.plot(traj_2[:,0],traj_2[:,1],color='orange')
        #     ax.plot(array, color="blue")
        # ax.set_title(f'Parameter {l+1}')
        # ax.set_xlabel('Time')
        # ax.set_ylim(0,1.3)
        # ax.set_ylabel('Difference')  

    # Adjust layout to prevent clipping of titles
    # plt.tight_layout()

    # Show the plot
    # plt.savefig("traj_compare.pdf")

def plot():
    distance_arrays = np.loadtxt("data/frechet_distance.csv", delimiter=",")
    plt.rcParams['font.family'] = 'Palatino'
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    for l in range(4):
        #print mean value of the normalized frechet distance for each parameter with std deviation
        print(f"Parameter {l+1} - Mean: {np.nanmean(distance_arrays[l,:])/583*100} - Std Dev: {np.nanstd(distance_arrays[l,:])/583*100}")

    for l in range(4):
        ax = axs[l // 2, l % 2]  # Get the current subplot
        ax.hist(distance_arrays[l,:]/583*100, bins=20, color="blue")
        ax.set_ylabel('Frequency',fontsize=21)
        ax.set_xlabel('Normalized Frechet Distance(%)',fontsize=21)
        ax.set_title(f'Parameter {l+1}',fontsize=21)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #increase the size of x and y ticks
        ax.tick_params(axis='both', which='major', labelsize=21)
        #also plot the mean value of the normalized frechet distance for each parameter
        # print(np.mean(distance_arrays[l,:]))
        ax.axvline(x=np.nanmean(distance_arrays[l,:])/583*100, color='r', linestyle='dashed', linewidth=2,label="Mean: "+str(round(np.nanmean(distance_arrays[l,:])/583*100,2)))
        ax.legend(fontsize=21, frameon=False)
        # plt.hist(distance_arrays[l,:], bins=25, color="blue")
        # plt.show()
    #set title to whole figure
    # fig.suptitle('Normalized Frechet Distance wrt total trajectory length')
    plt.tight_layout()
    plt.savefig("frechet_obs.pdf")
    plt.show()

if(__name__ == '__main__'):
    # main()
    plot()
