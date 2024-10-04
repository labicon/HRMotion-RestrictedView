import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math

def create_obsts_fixed(x_positions, y_positions):
    radius = []; circ_x = []; circ_y = []
    print(len(x_positions))
    for i in range(len(x_positions)):
        radius.append(random.randint(10, 10))
        circ_x.append((x_positions[i])*420/1000)
        circ_y.append((y_positions[i])*360/600)
    return [radius, circ_x, circ_y]

def plot_circle(x,y,r):
    theta = np.linspace(0,2*np.pi,100)
    x1 = r*np.cos(theta) + x
    y1 = r*np.sin(theta) + y
    return x1,y1

def read_traj(n):
    traj = np.loadtxt("data/positions_"+str(n)+".csv", delimiter=",")
    # obst = np.loadtxt("data/obstacles_"+str(n)+".csv", delimiter=",")

    return [traj,obst]


def angle_truth_value(robot_x, robot_y, robot_phi, obs_theta, circ_x, circ_y, r, distance_to_obstacle):
    relative_angle = math.atan2(circ_y - robot_y, circ_x - robot_x)
    truth_value = False
    if abs(relative_angle - robot_phi) <= obs_theta:
        truth_value = True
    # elif abs(relative_angle - robot_phi ) <= obs_theta + math.atan2(r, distance_to_obstacle):
    #     truth_value = True
    return truth_value

def plot_at_time(t):

    traj = np.loadtxt("data/bicycle_full_positions_expt.csv", delimiter=",")
    print(traj.shape)
    x_positions = [344.486, 483.842, 601.642, 871.514, 505.249, 606.248, 367.165]
    y_positions = [508.149, 299.161, 118.530, 362.297, 478.287, 403.393, 306.231]
    [radius, circ_x, circ_y] = create_obsts_fixed(x_positions, y_positions)

    obst = np.array([radius, circ_x, circ_y]).T

    #set font to Palatino
    plt.rcParams['font.family'] = 'Palatino'
    plt.figure()

    #plot circular obstacles with radius 10
    for i in range(obst.shape[0]):
        [x,y] = plot_circle(obst[i,1],obst[i,2],10)
        if i == 0:
            plt.plot(x,y,color='grey')
            plt.fill(x,y,'grey', label='Obstacles')
        else:
            plt.plot(x,y,color='grey')
            plt.fill(x,y,'grey')
    
    skirt_r =  100 # Sensor skirt radius
    obs_theta = math.pi/6
    
    #visualize sensor skirt which is triangle
    for i in [t]:
        x = traj[i,0]
        y = traj[i,1]
        phi = traj[i,4]
        x1 = x + skirt_r*np.cos(phi + obs_theta)
        y1 = y + skirt_r*np.sin(phi + obs_theta)
        x2 = x + skirt_r*np.cos(phi - obs_theta)
        y2 = y + skirt_r*np.sin(phi - obs_theta)
        # plt.plot([x,x1],[y,y1],color='moccasin')
        # plt.plot([x,x2],[y,y2],color='moccasin')
        # plt.plot([x1,x2],[y1,y2],color='moccasin')
        # #fill the triangle
        if i == t:
            plt.fill([x,x1,x2],[y,y1,y2],'moccasin',alpha=0.75,label='Sensor Skirt')
        else:
            plt.fill([x,x1,x2],[y,y1,y2],'moccasin',alpha=0.75)

        #if the obstacle is in the sensor skirt then color it red
        # print(obst.shape)
        for j in range(obst.shape[0]):
            x = traj[i,0]
            y = traj[i,1]
            phi = traj[i,4]
            distance_to_obstacle = np.sqrt((x-obst[j,1])**2 + (y-obst[j,2])**2)
            # print(distance_to_obstacle) 
            # print(obst[j,1],obst[j,2])
            
            relative_angle = math.atan2(obst[j,2] - y, obst[j,1] - x)
            if distance_to_obstacle <= skirt_r + 10 and angle_truth_value(x,y,phi,obs_theta,obst[j,1],obst[j,2],10,distance_to_obstacle):
                print("Obstacle in sensor skirt",obst[j,1],obst[j,2])
                print("distance to obstacle",distance_to_obstacle)
                [x_obs,y_obs] = plot_circle(obst[j,1],obst[j,2],10)
                # print(x,y)
                if i == t:
                    plt.plot(x_obs,y_obs,color='red')
                    plt.fill(x_obs,y_obs,'red',label='Detected Obstacles')
                else:
                    plt.plot(x_obs,y_obs,color='red')
                    plt.fill(x_obs,y_obs,'red')

        #visulaize the robot as a rectangle car
        x = traj[i,0]
        y = traj[i,1]
        theta = traj[i,2]
        # phi = traj[i,4]

        l = 40
        b = 20
        x1 = x + l/2 * np.cos(theta)
        y1 = y + l/2 * np.sin(theta)
        x2 = x - l/2 * np.cos(theta)
        y2 = y - l/2 * np.sin(theta)

        plt.plot([x1,x2],[y1,y2],color='orange',linewidth=10, label='Robot')

        # [x,y] = plot_circle(x,y,7)
        # if i == 1:
        #     plt.plot(x,y,color='orange')
        #     plt.fill(x,y,'orange',zorder=1,label='Robot')
        # else:
        #     plt.plot(x,y,color='orange',zorder=1)
        #     plt.fill(x,y,'orange',zorder=1)

    [x,y] = plot_circle(290,430,10)
    plt.plot(x,y,color='green')
    plt.fill(x,y,'green',label='Goal')

    plt.plot(traj[0:t+1,0],traj[0:t+1,1],color='blue',zorder=0,label='Trajectory')
    plt.plot(traj[t:,0],traj[t:,1],'--',color='blue',zorder=0,label='Final Trajectory')
    #axis off
    plt.axis('off')
    # plt.legend(frameon=False, loc='lower center', ncol=7, bbox_to_anchor=(0.5, -0.15))
    plt.axis('equal')
    plt.savefig("plots/expt_"+str(t)+".pdf",bbox_inches='tight')
    #plt.show()

if(__name__ == '__main__'):
    # main()
    times = [0,7,12,14]
    for t in times:
        plot_at_time(t)
