# want to visualize one of the trajectories

import numpy as np
import matplotlib.pyplot as plt
import math
import imageio

def read_traj(n):
    traj_1 = np.loadtxt("data_traj_compare/positions_1_"+str(n)+".csv", delimiter=",")
    traj_2 = np.loadtxt("data_traj_compare/positions_2_"+str(n)+".csv", delimiter=",")
    obst = np.loadtxt("data_traj_compare/obstacles_"+str(n)+".csv", delimiter=",")
    # obst = np.loadtxt("data/obstacles_"+str(n)+".csv", delimiter=",")

    return [traj_1,traj_2,obst]

def plot_circle(x,y,r):
    theta = np.linspace(0,2*np.pi,100)
    x1 = r*np.cos(theta) + x
    y1 = r*np.sin(theta) + y
    return x1,y1

def angle_truth_value(robot_x, robot_y, robot_phi, obs_theta, circ_x, circ_y, r, distance_to_obstacle):
    relative_angle = math.atan2(circ_y - robot_y, circ_x - robot_x)
    truth_value = False
    if abs(relative_angle - robot_phi) <= obs_theta:
        truth_value = True
    elif abs(relative_angle - robot_phi ) <= obs_theta + math.atan2(r, distance_to_obstacle):
        truth_value = True
    return truth_value

def main():
    [traj,obst] = read_traj(1)

    #set font to Palatino
    plt.rcParams['font.family'] = 'Palatino'

    #plot circular obstacles with radius 10
    for i in range(obst.shape[0]):
        [x,y] = plot_circle(obst[i,1],obst[i,2],10)
        if i == 0:
            plt.plot(x,y,color='grey')
            plt.fill(x,y,'grey', label='Obstacles')
        else:
            plt.plot(x,y,color='grey')
            plt.fill(x,y,'grey')

    # plt.Circle((obst[0,0],obst[0,1]),10,color='red',fill=False)

    skirt_r = 70   # Sensor skirt radius
    obs_theta = math.pi/6

    #visualize sensor skirt which is triangle
    for i in [70,800,1600,2500]:
        x = traj[i,0]
        y = traj[i,1]
        phi = traj[i,2]
        x1 = x + skirt_r*np.cos(phi + obs_theta)
        y1 = y + skirt_r*np.sin(phi + obs_theta)
        x2 = x + skirt_r*np.cos(phi - obs_theta)
        y2 = y + skirt_r*np.sin(phi - obs_theta)
        # plt.plot([x,x1],[y,y1],color='moccasin')
        # plt.plot([x,x2],[y,y2],color='moccasin')
        # plt.plot([x1,x2],[y1,y2],color='moccasin')
        # #fill the triangle
        if i == 70:
            plt.fill([x,x1,x2],[y,y1,y2],'moccasin',alpha=0.75,label='Sensor Skirt')
        else:
            plt.fill([x,x1,x2],[y,y1,y2],'moccasin',alpha=0.75)

        #if the obstacle is in the sensor skirt then color it red
        # print(obst.shape)
        for j in range(obst.shape[0]):
            x = traj[i,0]
            y = traj[i,1]
            phi = traj[i,2]
            distance_to_obstacle = np.sqrt((x-obst[j,1])**2 + (y-obst[j,2])**2)
            # print(distance_to_obstacle) 
            # print(obst[j,1],obst[j,2])
            
            relative_angle = math.atan2(obst[j,2] - y, obst[j,1] - x)
            if distance_to_obstacle <= skirt_r + 10 and angle_truth_value(x,y,phi,obs_theta,obst[j,1],obst[j,2],10,distance_to_obstacle):
                [x,y] = plot_circle(obst[j,1],obst[j,2],10)
                if i == 70:
                    plt.plot(x,y,color='red')
                    plt.fill(x,y,'red',label='Detected Obstacles')
                else:
                    plt.plot(x,y,color='red')
                    plt.fill(x,y,'red')
        
        #visulaize the robot
        [x,y] = plot_circle(x,y,7)
        if i == 70:
            plt.plot(x,y,color='orange')
            plt.fill(x,y,'orange',zorder=1,label='Robot')
        else:
            plt.plot(x,y,color='orange',zorder=1)
            plt.fill(x,y,'orange',zorder=1)

    [x,y] = plot_circle(600,400,10)
    plt.plot(x,y,color='green')
    plt.fill(x,y,'green',label='Goal')

    plt.plot(traj[:,0],traj[:,1],color='blue',zorder=0,label='Trajectory')
    #axis off
    plt.axis('off')
    plt.legend(frameon=False, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))
    plt.savefig("schematic.pdf")
    plt.show()


def creat_gif():
    # Create a gif of the trajectory
    images = []

    skirt_r = 70   # Sensor skirt radius
    obs_theta = math.pi/6
    # for t in 
    for trajs in [6,8,16]:
        [traj,traj_2,obst] = read_traj(trajs)
        length = traj.shape[0]
        for t in range(0,length,30):
            # fix figure size
            # fig = plt.figure(figsize=(10, 8))

            for i in range(obst.shape[0]):
                [x,y] = plot_circle(obst[i,1],obst[i,2],10)
                if i == 0:
                    plt.plot(x,y,color='grey')
                    plt.fill(x,y,'grey', label='Obstacles')
                else:
                    plt.plot(x,y,color='grey')
                    plt.fill(x,y,'grey')

            x = traj[t,0]
            y = traj[t,1]
            phi = traj[t,2]
            x1 = x + skirt_r*np.cos(phi + obs_theta)
            y1 = y + skirt_r*np.sin(phi + obs_theta)
            x2 = x + skirt_r*np.cos(phi - obs_theta)
            y2 = y + skirt_r*np.sin(phi - obs_theta)
            # plt.plot([x,x1],[y,y1],color='moccasin')
            # plt.plot([x,x2],[y,y2],color='moccasin')
            # plt.plot([x1,x2],[y1,y2],color='moccasin')
            # #fill the triangle
            # if t == 0:
            plt.fill([x,x1,x2],[y,y1,y2],'moccasin',alpha=0.75,label='Sensor Skirt')
            # else:
            #     plt.fill([x,x1,x2],[y,y1,y2],'moccasin',alpha=0.75)

            #if the obstacle is in the sensor skirt then color it red
            # print(obst.shape)
            for j in range(obst.shape[0]):
                x = traj[t,0]
                y = traj[t,1]
                phi = traj[t,2]
                distance_to_obstacle = np.sqrt((x-obst[j,1])**2 + (y-obst[j,2])**2)
                # print(distance_to_obstacle) 
                # print(obst[j,1],obst[j,2])
                
                relative_angle = math.atan2(obst[j,2] - y, obst[j,1] - x)
                if distance_to_obstacle <= skirt_r + 10 and angle_truth_value(x,y,phi,obs_theta,obst[j,1],obst[j,2],10,distance_to_obstacle):
                    [x,y] = plot_circle(obst[j,1],obst[j,2],10)
                    plt.plot(x,y,color='red')
                    plt.fill(x,y,'red',label='Detected Obstacles')
                    plt.plot(x,y,color='red')
                    plt.fill(x,y,'red')

            #visulaize the robot
            [x,y] = plot_circle(x,y,7)
            plt.plot(x,y,color='orange')
            plt.fill(x,y,'orange',zorder=1,label='Robot')

            #visualize the goal
            [x,y] = plot_circle(600,400,10)
            plt.plot(x,y,color='green')
            plt.fill(x,y,'green',label='Goal')

            plt.plot(traj[0:t,0],traj[0:t,1],color='blue',zorder=0,label='Trajectory')
            #axis off
            plt.axis('off')
            plt.xlim(-10,670)
            plt.ylim(-60,500)
            # make axis ascpet ratio 1
            plt.gca().set_aspect('equal', adjustable='box')
            # plt.legend(frameon=False, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))

            plt.axis('off')
            plt.tight_layout()
            plt.savefig("gifs/images/"+str(i)+".png")
            images.append(imageio.imread("gifs/images/"+str(i)+".png"))
            plt.close()
    
    #make a fast gif of the trajectory
    imageio.mimsave('gifs/trajectory.gif', images,fps=30)


def creat_gif_both_agents_together():
    # Create a gif of the trajectory
    images = []

    skirt_r = 70   # Sensor skirt radius
    obs_theta = math.pi/6
    # for t in 
    for trajs in [6,16,25]:
        [traj_1,traj_2,obst] = read_traj(trajs)
        # length is minimum of the two trajectories
        length = min(traj_1.shape[0], traj_2.shape[0])

        plt.rcParams['font.family'] = 'Palatino'

        for t in range(0,length,30):
            # fix figure size
            fig = plt.figure(figsize=(10, 16))

            for i in range(obst.shape[0]):
                [x,y] = plot_circle(obst[i,1],obst[i,2],10)
                if i == 0:
                    plt.plot(x,y,color='grey')
                    plt.fill(x,y,'grey', label='Obstacles')
                else:
                    plt.plot(x,y,color='grey')
                    plt.fill(x,y,'grey')

            x = traj_1[t,0]
            y = traj_1[t,1]
            phi = traj_1[t,2]
            x1 = x + skirt_r*np.cos(phi + obs_theta)
            y1 = y + skirt_r*np.sin(phi + obs_theta)
            x2 = x + skirt_r*np.cos(phi - obs_theta)
            y2 = y + skirt_r*np.sin(phi - obs_theta)
            # plt.plot([x,x1],[y,y1],color='moccasin')
            # plt.plot([x,x2],[y,y2],color='moccasin')
            # plt.plot([x1,x2],[y1,y2],color='moccasin')
            # #fill the triangle
            # if t == 0:
            plt.fill([x,x1,x2],[y,y1,y2],'moccasin',alpha=0.75)
            # else:
            #     plt.fill([x,x1,x2],[y,y1,y2],'moccasin',alpha=0.75)

            # plot the sensor skirt for the second agent
            x = traj_2[t,0]
            y = traj_2[t,1]
            phi = traj_2[t,2]
            x1 = x + skirt_r*np.cos(phi + obs_theta)
            y1 = y + skirt_r*np.sin(phi + obs_theta)
            x2 = x + skirt_r*np.cos(phi - obs_theta)
            y2 = y + skirt_r*np.sin(phi - obs_theta)
            plt.fill([x,x1,x2],[y,y1,y2],'moccasin',alpha=0.75,label='Sensor Skirt')

            #if the obstacle is in the sensor skirt then color it red
            # print(obst.shape)
            count = 0
            for j in range(obst.shape[0]):
                x = traj_1[t,0]
                y = traj_1[t,1]
                phi = traj_1[t,2]
                distance_to_obstacle = np.sqrt((x-obst[j,1])**2 + (y-obst[j,2])**2)
                # print(distance_to_obstacle) 
                # print(obst[j,1],obst[j,2])
                
                relative_angle = math.atan2(obst[j,2] - y, obst[j,1] - x)
                if distance_to_obstacle <= skirt_r + 10 and angle_truth_value(x,y,phi,obs_theta,obst[j,1],obst[j,2],10,distance_to_obstacle):
                    count += 1
                    [x,y] = plot_circle(obst[j,1],obst[j,2],10)
                    plt.plot(x,y,color='red')
                    plt.fill(x,y,'red')
                    plt.plot(x,y,color='red')
                    plt.fill(x,y,'red')
            
            #if the obstacle is in the sensor skirt then color it red
            # print(obst.shape)
            for j in range(obst.shape[0]):
                x = traj_2[t,0]
                y = traj_2[t,1]
                phi = traj_2[t,2]
                distance_to_obstacle = np.sqrt((x-obst[j,1])**2 + (y-obst[j,2])**2)
                # print(distance_to_obstacle) 
                # print(obst[j,1],obst[j,2])
                
                relative_angle = math.atan2(obst[j,2] - y, obst[j,1] - x)
                if distance_to_obstacle <= skirt_r + 10 and angle_truth_value(x,y,phi,obs_theta,obst[j,1],obst[j,2],10,distance_to_obstacle):
                    count += 1
                    [x,y] = plot_circle(obst[j,1],obst[j,2],10)
                    plt.plot(x,y,color='red')
                    plt.fill(x,y,'red')
                    plt.plot(x,y,color='red')
                    plt.fill(x,y,'red')

            # if count is zero then add a dummy obstacle for label consisentcy
            # print(count)
            # if count == 0:
            [x,y] = plot_circle(-100,-100,10)
            plt.plot(x,y,color='red')
            plt.fill(x,y,'red',label='Detected Obstacles')
            plt.plot(x,y,color='red')
            plt.fill(x,y,'red')

            #visulaize the robot
            x = traj_1[t,0]
            y = traj_1[t,1]
            [x,y] = plot_circle(x,y,7)
            plt.plot(x,y,color='orange')
            plt.fill(x,y,'orange',zorder=1,label='Actual Agent')

            #visulaize the robot
            x = traj_2[t,0]
            y = traj_2[t,1]
            [x,y] = plot_circle(x,y,7)
            plt.plot(x,y,color='purple')
            plt.fill(x,y,'purple',zorder=1,label='Learned Policy')

            #visualize the goal
            [x,y] = plot_circle(600,400,10)
            plt.plot(x,y,color='green')
            plt.fill(x,y,'green',label='Goal')
            
            # plot the trajectory of the first agent
            plt.plot(traj_1[0:t,0],traj_1[0:t,1],color='blue',zorder=0,label='Actual Agent Trajectory')
            # plot the trajectory of the second agent
            plt.plot(traj_2[0:t,0],traj_2[0:t,1],color='purple',zorder=0,label='Learned Policy Trajectory')
            #axis off
            plt.axis('off')
            plt.xlim(-10,670)
            plt.ylim(-60,500)
            # make axis ascpet ratio 1
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend(frameon=False, loc='center right', bbox_to_anchor=(1.5, 0.5))

            plt.axis('off')
            # plt.tight_layout()
            plt.savefig("gifs/images/"+str(t)+".pdf",bbox_inches='tight')
            images.append(imageio.imread("gifs/images/"+str(t)+".png"))
            plt.close()

    #make a fast gif of the trajectory
    imageio.mimsave('gifs/trajectory.gif', images,fps=30)         
            
    

if(__name__ == '__main__'):
    # main()
    # creat_gif()
    creat_gif_both_agents_together()