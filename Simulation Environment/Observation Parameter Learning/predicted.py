import numpy as np
import pygame
from pygame.locals import *
from robot import *
import random

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

def main():
    # PyGame inits
    pygame.init()
    pygame.display.set_caption('Unicycle robot')
    clock = pygame.time.Clock()
    ticks = pygame.time.get_ticks()

    for n in range(500):

        frames = 0

        # Robot
        robot_x = 100; robot_y = 100; robot_phi = 0; robot_l = 15; robot_b = 6  # Initial position
        position_array = np.array([[robot_x, robot_y, robot_phi]])
        skirt_r = 97.4  # Sensor skirt radius
        obs_theta = 0.7625
        goalX = np.array([600, 400])    # goal position

        data = {"screen":screen, "goalX":goalX, "vmax":0.5, "gtg_scaling":0.0001, "K_p":0.01, "ao_scaling":0.00005}

        # Create obstacles
        # [radius, circ_x, circ_y] = create_circular_obsts(num_circ_obsts)
        #
        # obstacles = np.array([radius, circ_x, circ_y]).T

        obstacles = np.loadtxt("data/obstacles_"+str(n)+".csv", delimiter=",")

        num_circ_obsts = obstacles.shape[0]
        radius = obstacles[:,0]
        circ_x = obstacles[:,1]
        circ_y = obstacles[:,2]

        # PyGame loop
        while(1):
            # To exit
            event = pygame.event.poll()
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                break
            screen.fill((50, 55, 60))   # background

            # Draw robot, sensor skirt, obstacles and goal
            bot = robot(robot_x, robot_y, robot_phi, robot_l, robot_b, data)
            #pygame.draw.circle(screen, (100, 100, 100), (int(bot.x), int(bot.y)), skirt_r, 0)   # Draw sensor skirt
            pygame.draw.polygon(screen, (100,100,100), [(int(bot.x), int(bot.y)),
                                                        (int(bot.x + skirt_r*math.cos(bot.phi + obs_theta)), int(bot.y + skirt_r*math.sin(bot.phi + obs_theta))) ,
                                                        (int(bot.x + skirt_r*math.cos(bot.phi - obs_theta)), int(bot.y + skirt_r*math.sin(bot.phi - obs_theta)))])
            draw_circular_obsts(num_circ_obsts, radius, circ_x, circ_y, (0, 0, 255))
            bot.show()    # Draw the robot
            pygame.draw.circle(screen, (0,255,0), goalX, 8, 0)  # Draw goal

            # Check if obstacles are in sensor skirt
            close_obst = []; dist = []
            close_radius = []
            close_circ_x = []
            close_circ_y = []
            for i in range(num_circ_obsts):
                distance = math.sqrt((circ_x[i] - robot_x)**2 + (circ_y[i] - robot_y)**2)
                if( distance <= (skirt_r + radius[i]) and angle_truth_value(robot_x, robot_y, robot_phi, obs_theta, circ_x[i], circ_y[i],radius[i],distance)):
                    if random.random() <= 0.75:
                        close_radius.append(radius[i])
                        close_circ_x.append(circ_x[i])
                        close_circ_y.append(circ_y[i])
                        close_obst.append([circ_x[i], circ_y[i], radius[i]])
                        dist.append(distance)
            # Go to goal
            if(len(close_obst) == 0) or (math.sqrt((goalX[0] - robot_x)**2 + (goalX[1] - robot_y)**2) <= min(dist)):           # No obstacle in sensor skirt
                draw_circular_obsts(len(close_obst), close_radius, close_circ_x, close_circ_y, (255, 0, 0))
                [v, omega] = bot.go_to_goal()   # output from controller go_to_goal()
            # Paranoid behavior - run away from obstacle
            else:
                #print(close_circ_x)
                draw_circular_obsts(len(close_obst), close_radius, close_circ_x, close_circ_y, (255, 0, 0))
                closest_obj = dist.index(min(dist)) # gives the index of the closest object
                obs_radius = 10
                if(len(close_obst) > 1):
                    obs_radius = 10
                obs_x = np.mean(close_circ_x)
                obs_y = np.mean(close_circ_y)
                #obstX = np.array([circ_x[closest_obj], circ_y[closest_obj]])
                obstX = np.array([obs_x, obs_y])
                [v, omega] = bot.avoid_obst(obstX, obs_radius)

            # Update robot position and orientation as per control input
            robot_x += v*math.cos(robot_phi); robot_y+= v*math.sin(robot_phi); robot_phi += omega; robot_phi = (robot_phi + np.pi) % (2 * np.pi) - np.pi
            position_array = np.append(position_array, np.array([[robot_x,robot_y,robot_phi]]),axis = 0)

            # FPS. Print if required
            clock.tick(300)     # To limit fps, controls speed of the animation
            fps = (frames*1000)/(pygame.time.get_ticks() - ticks)   # calculate current fps

            # Update PyGame display
            pygame.display.flip()
            frames+=1

            distance_to_goal = math.sqrt((goalX[0] - robot_x)**2 + (goalX[1] - robot_y)**2)
            threshold = 15
            if distance_to_goal < threshold or frames > 5000:
                if n%20 == 0:
                    print("Trajectory number:",n+1)
                np.savetxt("data/positions_pred_4"+str(n)+".csv", position_array, delimiter=",",fmt ='% s')
                #np.savetxt("data/obstacles_"+str(n)+".csv", obstacles, delimiter=",",fmt ='% s')
                break

if(__name__ == '__main__'):
    main()
