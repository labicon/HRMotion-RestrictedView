import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import os, sys
import numpy as np

import pygame
from pygame.locals import *
from robot import *
import random
from bayes_opt import BayesianOptimization

# Screen
screen_width = 640; screen_height = 480
screen = pygame.display.set_mode([screen_width, screen_height], DOUBLEBUF)

# Obstacles
num_circ_obsts = 30; obst_min_radius = 10; obst_max_radius = 10  # for circular obstacles

def create_circular_obsts(num):
    radius = []; circ_x = []; circ_y = []
    for i in range(num):
        radius.append(random.randint(obst_min_radius, obst_max_radius))
        circ_x.append(random.randint(radius[i], screen_width - radius[i]))
        circ_y.append(random.randint(radius[i], screen_height - radius[i]))
    return [radius, circ_x, circ_y]

def draw_circular_obsts(num, radius, circ_x, circ_y, color):
    for i in range(num):
        pygame.draw.circle(screen, color, (circ_x[i], circ_y[i]), radius[i], 0)

def angle_truth_value(robot_x, robot_y, robot_phi, obs_theta, circ_x, circ_y, r, distance_to_obstacle):
    relative_angle = math.atan2(circ_y - robot_y, circ_x - robot_x)
    truth_value = False
    if abs(relative_angle - robot_phi) <= obs_theta:
        truth_value = True
    elif abs(relative_angle - robot_phi ) <= obs_theta + math.atan2(r, distance_to_obstacle):
        truth_value = True
    return truth_value

def prob_trajectory(current_tra,obstacles, goalX, skirt_r, obs_theta, epsilone):

    l = current_tra.shape[0]

    robot_l = 15; robot_b = 6  # Initial position
    #position_array = np.array([[robot_x, robot_y, robot_phi]])

    data = {"screen":screen, "goalX":goalX, "vmax":0.5, "gtg_scaling":0.0001, "K_p":0.01, "ao_scaling":0.00005}
    #probability_matrix = create_probability_matrix(grid_size, agent_position, A, mu, sigma)

    probs = np.ones(l-1)

    for i in range(l-1):
        bot = robot(current_tra[i,0], current_tra[i,1], current_tra[i,2], robot_l, robot_b, data)

        if math.cos(current_tra[i,2]) > 0.5:
            v = (current_tra[i+1,0] - current_tra[i,0])/math.cos(current_tra[i,2])
            omega = current_tra[i+1,2] - current_tra[i,2]
        else:
            v = (current_tra[i+1,1] - current_tra[i,1])/math.sin(current_tra[i,2])
            omega = current_tra[i+1,2] - current_tra[i,2]

        # Check if obstacles are in sensor skirt
        num_circ_obsts = obstacles.shape[0]
        radius = obstacles[:,0]
        circ_x = obstacles[:,1]
        circ_y = obstacles[:,2]
        close_obst = []; dist = []
        close_radius = []
        close_circ_x = []
        close_circ_y = []
        for o in range(num_circ_obsts):
            distance = math.sqrt((circ_x[o] - current_tra[i,0])**2 + (circ_y[o] - current_tra[i,1])**2)
            if( distance <= (skirt_r + radius[o]) and angle_truth_value(current_tra[i,0], current_tra[i,1], current_tra[i,2], obs_theta, circ_x[o], circ_y[o],radius[o],distance)):
                close_radius.append(radius[o])
                close_circ_x.append(circ_x[o])
                close_circ_y.append(circ_y[o])
                close_obst.append([circ_x[o], circ_y[o], radius[o]])
                dist.append(distance)

        [v_act, omega_act] = bot.go_to_goal()

        if abs(v_act-v)+abs(omega_act-omega) > epsilone:
            if(len(close_obst) == 0) or (math.sqrt((goalX[0] - current_tra[i,0])**2 + (goalX[1] - current_tra[i,1])**2) <= min(dist)):
                probs[i] = 0.00001
            else:
                probs[i] = 0.75

        else:
            if(len(close_obst) == 0) or (math.sqrt((goalX[0] - current_tra[i,0])**2 + (goalX[1] - current_tra[i,1])**2) <= min(dist)):
                probs[i] = 1
            else:
                probs[i] = 0.25

        # if(len(close_obst) == 0) or (math.sqrt((goalX[0] - current_tra[i,0])**2 + (goalX[1] - current_tra[i,1])**2) <= min(dist)):
        #     continue
        # else:
        #     [v_act, omega_act] = bot.go_to_goal()
        #     if abs(v_act-v)+abs(omega_act-omega) < epsilone:
        #         #print(abs(v_act-v)+abs(omega_act-omega))
        #         probs[i] = 0.25
        #     else:
        #         probs[i] = 0.75

    return np.sum(np.log(probs))

def black_box_function(x,y,l):
    #traj_i_array = np.array([170,  27, 159, 135,   3,  17,  59, 140,  98, 114])
    #traj_i_array = np.array([5])
    sigma = x
    epsilone = 1e-1
    total_prob = 0
    goalX = np.array([600, 400])
    skirt_r = x
    obs_theta = y
    for i in range(50):
        current_tra =  np.loadtxt("data/orig_traj/positions_param_"+str(l)+"_"+str(i)+".csv", delimiter=",")
        obstacles = np.loadtxt("data/orig_traj/obstacles_"+str(l)+"_"+str(i)+".csv", delimiter=",")
        total_prob += prob_trajectory(current_tra,obstacles, goalX, skirt_r, obs_theta, epsilone)

    return total_prob

def fun(x, y):
    return x**2 + y

##### TO CREATE A SERIES OF PICTURES

def make_views(ax,angles,elevation=None, width=10, height = 15,
                prefix='tmprot_',**kwargs):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created.

    Returns: the list of files created (for later removal)
    """

    files = []
    ax.figure.set_size_inches(width,height)

    for i,angle in enumerate(angles):

        ax.view_init(elev = elevation, azim=angle)
        fname = '%s%03d.jpeg'%(prefix,i)
        ax.figure.savefig(fname)
        files.append(fname)

    return files



##### TO TRANSFORM THE SERIES OF PICTURE INTO AN ANIMATION

def make_movie(files,output, fps=10,bitrate=1800,**kwargs):
    """
    Uses mencoder, produces a .mp4/.ogv/... movie from a list of
    picture files.
    """

    output_name, output_ext = os.path.splitext(output)
    command = { '.mp4' : 'mencoder "mf://%s" -mf fps=%d -o %s.mp4 -ovc lavc\
                         -lavcopts vcodec=msmpeg4v2:vbitrate=%d'
                         %(",".join(files),fps,output_name,bitrate)}

    command['.ogv'] = command['.mp4'] + '; ffmpeg -i %s.mp4 -r %d %s'%(output_name,fps,output)

    # print command[output_ext]
    output_ext = os.path.splitext(output)[1]
    os.system(command[output_ext])



def make_gif(files,output,delay=100, repeat=True,**kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """

    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'
              %(delay,loop," ".join(files),output))




def make_strip(files,output,**kwargs):
    """
    Uses imageMagick to produce a .jpeg strip from a list of
    picture files.
    """

    os.system('montage -tile 1x -geometry +0+0 %s %s'%(" ".join(files),output))



##### MAIN FUNCTION

def rotanimate(ax, angles, output, **kwargs):
    """
    Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
    a 3D ax

    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """

    output_ext = os.path.splitext(output)[1]

    files = make_views(ax,angles, **kwargs)

    D = { '.mp4' : make_movie,
          '.ogv' : make_movie,
          '.gif': make_gif ,
          '.jpeg': make_strip,
          '.png':make_strip}

    D[output_ext](files,output,**kwargs)

    for f in files:
        os.remove(f)

def main():

    params = np.array([[55,0.785],[99,1.571],[22,1.047],[70,0.524]])
    params_pred = np.array([[55.11, 0.815],[99.94, 1.17],[22.09, 1.155],[70, 0.524]])

    n_params = params.shape[0]
    print("Number of parameters:",n_params)

    for l in range(n_params):
        X = np.loadtxt("data/X_"+str(l)+".csv", delimiter=",")
        Y = np.loadtxt("data/Y_"+str(l)+".csv", delimiter=",")
        Z = np.loadtxt("data/Z_"+str(l)+".csv", delimiter=",")

        # print(Z)
        z = Z.max()
        x,y = np.unravel_index(Z.argmax(), Z.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z.T/abs(z), zorder=3, color='orange',alpha=0.4)

        ax.scatter(X[0,x], Y[y,0], z/abs(z), color='red', s=100, zorder=2)

        ax.text(X[0,x], Y[y,0], z/abs(z) + 1,  'skirt_r = %s, theta = %s2 - grid search' % (str(round(X[0,x],3)), str(round(Y[y,0], 4))) , size=20, zorder=1,
        color='k')

        ax.scatter(params[l,0], params[l,1], black_box_function(params[l,0],params[l,0],l)/abs(z), color='purple', s=100, zorder=1)

        ax.text(params[l,0], params[l,1], black_box_function(params[l,0],params[l,0],l)/abs(z)+6,  'skirt_r = %s, theta = %s2 - real' % (str(round(params[l,0],3)), str(round(params[l,1], 4))) , size=20, zorder=1,
        color='purple')

        ax.scatter(params_pred[l,0], params_pred[l,1], black_box_function(params_pred[l,0],params_pred[l,0],l)/abs(z), color='green', s=100, zorder=1)

        ax.text(params_pred[l,0], params_pred[l,1], black_box_function(params_pred[l,0],params_pred[l,0],l)/abs(z)+7,  'skirt_r = %s, theta = %s2 - cross entropy' % (str(round(params_pred[l,0],3)), str(round(params_pred[l,1], 4))) , size=20, zorder=1,
        color='green')

        ax.set_xlabel('skirt r',size=15)
        ax.set_ylabel('theta',size=15)
        ax.set_zlabel('probability value',size=15)
        ax.set_title('Surface plot for probability of trajectories',size=25)

        angles = np.linspace(0,360,51)[:-1] # Take 50 angles between 0 and 360

        # create an animated gif (20ms between frames)
        rotanimate(ax, angles,'grid_search_list'+str(l)+'.gif',delay=20)

if(__name__ == '__main__'):
    main()
