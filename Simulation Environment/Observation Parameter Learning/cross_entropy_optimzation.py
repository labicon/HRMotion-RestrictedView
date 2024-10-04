import numpy as np
import pygame
from pygame.locals import *
import random
import math

# Assuming the robot module is available
from robot import *

# Screen setup
screen_width = 640
screen_height = 480
screen = pygame.display.set_mode([screen_width, screen_height], DOUBLEBUF)

# Obstacles
num_circ_obsts = 30
obst_min_radius = 10
obst_max_radius = 10

def create_circular_obsts(num):
    radius = []
    circ_x = []
    circ_y = []
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
    elif abs(relative_angle - robot_phi) <= obs_theta + math.atan2(r, distance_to_obstacle):
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


def black_box_function(params):
    #traj_i_array = np.array([170,  27, 159, 135,   3,  17,  59, 140,  98, 114])
    #traj_i_array = np.array([5])
    x, y = params
    epsilone = 1e-1
    total_prob = 0
    goalX = np.array([600, 400])
    skirt_r = x
    obs_theta = y
    for i in range(30):
        current_tra =  np.loadtxt("data/positions_"+str(i)+".csv", delimiter=",")
        obstacles = np.loadtxt("data/obstacles_"+str(i)+".csv", delimiter=",")
        total_prob += prob_trajectory(current_tra,obstacles, goalX, skirt_r, obs_theta, epsilone)

    return total_prob/500

def cross_entropy_method(obj_func, bounds, num_iterations, sample_size, elite_frac):
    mean = np.array([np.mean(bound) for bound in bounds])
    std_dev = np.array([(bound[1] - bound[0]) / 2 for bound in bounds])
    print("Start")
    for iteration in range(num_iterations):
        samples = np.random.normal(mean, std_dev, (sample_size, len(bounds)))
        samples = np.clip(samples, [b[0] for b in bounds], [b[1] for b in bounds])
        print('before the black box')
        sample_scores = np.array([obj_func(sample) for sample in samples])
        print('after the black box')
        elite_idxs = sample_scores.argsort()[-int(sample_size * elite_frac):]
        elite_samples = samples[elite_idxs]

        mean = np.mean(elite_samples, axis=0)
        std_dev = np.std(elite_samples, axis=0)
        print(f"The current parameter is {mean}")
        # Print progress
        print(f"Iteration {iteration + 1}/{num_iterations}, Current Best Score: {sample_scores[elite_idxs[-1]]}")

    return mean


def main():
    # PyGame inits (uncomment these if running the simulation)
    # pygame.init()
    # pygame.display.set_caption('Unicycle robot')
    # screen = pygame.display.set_mode([640, 480], DOUBLEBUF)

    bounds = [(1, 100), (0, math.pi)]
    best_params = cross_entropy_method(
        black_box_function, bounds, num_iterations=10, sample_size=50, elite_frac=0.2)

    print("Best Parameters:", best_params)
    # Additional code to use best_params in your simulation

if __name__ == '__main__':
    main()
