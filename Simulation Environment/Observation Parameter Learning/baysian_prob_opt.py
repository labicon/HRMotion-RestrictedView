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

def prob_trajectory(current_tra,obstacles, goalX, p1, epsilone):

    skirt_r = 70
    obs_theta = math.pi//6

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
                probs[i] = p1

        else:
            if(len(close_obst) == 0) or (math.sqrt((goalX[0] - current_tra[i,0])**2 + (goalX[1] - current_tra[i,1])**2) <= min(dist)):
                probs[i] = 1
            else:
                probs[i] = 1-p1

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

def black_box_function0(x):
    #traj_i_array = np.array([170,  27, 159, 135,   3,  17,  59, 140,  98, 114])
    #traj_i_array = np.array([5])
    # sigma = x
    epsilone = 1e-1
    total_prob = 0
    goalX = np.array([600, 400])
    p1 = x
    for i in range(500):
        current_traj = np.loadtxt("prob_data/prob_0.2/positions_"+str(i)+".csv", delimiter=",")
        obstacles = np.loadtxt("prob_data/prob_0.2/obstacles_"+str(i)+".csv", delimiter=",")
        total_prob += prob_trajectory(current_traj,obstacles, goalX, p1, epsilone)

    return total_prob/500

def black_box_function1(x):
    #traj_i_array = np.array([170,  27, 159, 135,   3,  17,  59, 140,  98, 114])
    #traj_i_array = np.array([5])
    # sigma = x
    epsilone = 1e-1
    total_prob = 0
    goalX = np.array([600, 400])
    p1 = x
    for i in range(500):
        current_traj = np.loadtxt("prob_data/prob_0.4/positions_"+str(i)+".csv", delimiter=",")
        obstacles = np.loadtxt("prob_data/prob_0.4/obstacles_"+str(i)+".csv", delimiter=",")
        total_prob += prob_trajectory(current_traj,obstacles, goalX, p1, epsilone)

    return total_prob/500

def black_box_function2(x):
    #traj_i_array = np.array([170,  27, 159, 135,   3,  17,  59, 140,  98, 114])
    #traj_i_array = np.array([5])
    # sigma = x
    epsilone = 1e-1
    total_prob = 0
    goalX = np.array([600, 400])
    p1 = x
    for i in range(500):
        current_traj = np.loadtxt("prob_data/prob_0.6/positions_"+str(i)+".csv", delimiter=",")
        obstacles = np.loadtxt("prob_data/prob_0.6/obstacles_"+str(i)+".csv", delimiter=",")
        total_prob += prob_trajectory(current_traj,obstacles, goalX, p1, epsilone)

    return total_prob/500

def black_box_function3(x):
    #traj_i_array = np.array([170,  27, 159, 135,   3,  17,  59, 140,  98, 114])
    #traj_i_array = np.array([5])
    # sigma = x
    epsilone = 1e-1
    total_prob = 0
    goalX = np.array([600, 400])
    p1 = x
    for i in range(500):
        current_traj = np.loadtxt("prob_data/prob_0.8/positions_"+str(i)+".csv", delimiter=",")
        obstacles = np.loadtxt("prob_data/prob_0.8/obstacles_"+str(i)+".csv", delimiter=",")
        total_prob += prob_trajectory(current_traj,obstacles, goalX, p1, epsilone)

    return total_prob/500

# def black_box_function(x):
#     #traj_i_array = np.array([170,  27, 159, 135,   3,  17,  59, 140,  98, 114])
#     #traj_i_array = np.array([5])
#     sigma = x
#     epsilone = 1e-1
#     total_prob = 0
#     goalX = np.array([600, 400])
#     p1 = x
#     for i in range(500):
#         current_traj = np.loadtxt("prob_data/prob+"/positions_"+str(i)+".csv", delimiter=",")
#         obstacles = np.loadtxt("prob_data/prob+"+str(p_value)+"/obstacles_"+str(i)+".csv", delimiter=",")
#         total_prob += prob_trajectory(current_traj,obstacles, goalX, p1, epsilone)

#     return total_prob/500

def main():
    # PyGame inits
    # pygame.init()
    # pygame.display.set_caption('Unicycle robot')
    # clock = pygame.time.Clock()
    # ticks = pygame.time.get_ticks()

    pbounds0 = {'x': (0.1, 0.5)}
    pbounds1 = {'x': (0.15, 0.65)}
    pbounds2 = {'x': (0.4, 0.9)}
    pbounds3 = {'x': (0.5, 0.95)}

    pbounds_list = [pbounds0, pbounds1, pbounds2, pbounds3]

    black_box_functions = [black_box_function0, black_box_function1, black_box_function2, black_box_function3]

    predictions = np.array([0.2209, 0.4352, 0.6140, 0.8249])

    for i in [3]:
        optimizer = BayesianOptimization(
            f=black_box_functions[i],
            pbounds=pbounds_list[i],
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )
        optimizer.maximize(
            init_points=4,
            n_iter=45,
        )

        predictions[i] = optimizer.max['params']['x']

    #prob = black_box_function(70,math.pi/6)

    # prob = black_box_function(0.75)

    # print(prob)

    np.savetxt("prob_data/predictions.csv", predictions, delimiter=",")

if(__name__ == '__main__'):
    main()