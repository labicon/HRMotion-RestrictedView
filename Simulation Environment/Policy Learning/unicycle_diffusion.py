## First let's write down the diffusion process

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.animation as animation
import torch.nn as nn
import random

import pygame
from pygame.locals import *
from unicycle_robot import *

#real distribution

#load data

trajectory = np.loadtxt("data/positions.csv", delimiter=",", dtype=float)
controls = np.loadtxt("data/controls.csv", delimiter=",", dtype=float)
obstacles = np.loadtxt("data/obstacles.csv", delimiter=",", dtype=float)

combined = np.concatenate((controls,trajectory[0:-1,:], obstacles), axis=1)

class Denoiser(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9+1, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, 2) )
        
    def forward(self, x, t):
        # s = t*torch.ones_like(x)
        s = t*torch.ones((x.shape[0], 1))
        return self.net( torch.cat((x, s), dim=1) )

denoiser = Denoiser(64)

# mu_data = 1.
# sigma_data = 0.01

nb_epochs = 10000
batch_size = 128
lr = 1e-3

losses = np.zeros(nb_epochs)
optimizer = torch.optim.Adam(denoiser.parameters(), lr)

dim = 2000

#noise scales

# betas = torch.tensor([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.999])
betas = torch.tensor([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.999])

alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, 0)


def qs(x0, alphas_bar):
    return [stats.norm(loc = torch.sqrt(alphas_bar[i])*x0, scale = torch.sqrt(1-alphas_bar[i])*torch.eye(len(x0)) ) for i in range(len(alphas_bar))]

for epoch in range(nb_epochs):
    # x0 = stats.norm(loc = mu_data, scale = np.sqrt(sigma_data)).rvs(size=batch_size)
    # x0 = torch.tensor(x0).float().view(-1,1)
    t = random.randint(0, len(alphas_bar)-1)

    x0 = torch.tensor(np.array(random.choices(combined, k=batch_size))).float().view(-1, combined.shape[1])

    # x0[:, :2] += torch.randn_like(x0[:, :2])

    eps = torch.randn_like(x0[:,:2])

    x_noised  = x0
    x_noised[:, :2] = np.sqrt(alphas_bar[t])*x0[:,:2] + np.sqrt(1-alphas_bar[t])*eps

    pred = denoiser(x_noised, alphas[t])

    loss = torch.linalg.vector_norm(eps - pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses[epoch] = loss.detach().item()
    
plt.title("Training loss")
plt.plot(np.arange(nb_epochs), losses)
# plt.ylim(0, 100)
plt.show()

def plot_sample(x, id_noise_scale):
    plt.title(f"Noised scale {id_noise_scale}")
    plt.scatter(np.arange(dim), x.numpy())
    plt.ylim([-3., 3.])
    plt.show()

torch.save(denoiser.state_dict(), 'checkpoints/diffusion_checkpoint_new.pth')

def compute_action_diff(alphas_bar, alphas, betas, denoiser,state,obstacle):
    alpha_bar = torch.prod(1 - betas)
    u0 = torch.randn((1,2))*torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar)

    # print(np.shape(u0))
    # print(np.shape(state))
    # print(np.shape(obstacle))

    x = torch.cat((u0,state, obstacle), dim=1)

    for t in range(len(alphas_bar),0,-1):
        if t>1:
            z = torch.randn_like(u0) 
        else:
            z = 0
        sigma_sq = betas[t-1] * (1 - alphas_bar[t-1]/alphas[t-1])/(1 - alphas_bar[t-1])
        with torch.no_grad():
            # print(np.shape((1/np.sqrt(alphas[t-1]))*(x[:,0:2] - (1-alphas[t-1])* denoiser(x, 1-betas[t-1])/(np.sqrt(1-alphas_bar[t-1])))))
            # print(np.shape(torch.sqrt(sigma_sq)*z))
            # print(np.shape(x[:,0:2]))
            x[:,0:2] = (1/np.sqrt(alphas[t-1]))*(x[:,0:2] - (1-alphas[t-1])* denoiser(x, 1-betas[t-1])/(np.sqrt(1-alphas_bar[t-1]))) + torch.sqrt(sigma_sq)*z
    return x[:,0:2]


x = torch.tensor(np.array(random.choices(combined, k=1))).float().view(-1, combined.shape[1])

state = x[:, 2:4]
obstacle = x[:, 4:]

print("Actial action:",state[:,0:2])
print("Predicted action:",compute_action_diff(alphas_bar, alphas, betas, denoiser,state,obstacle))

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

    frames = 0
    run = 0
    old_frame = 0

    print(run)
    # Robot
    robot_x = 100; robot_y = 100; robot_phi = 0; robot_l = 15; robot_b = 6  # Initial position
    if run == 0:
        position_array = np.array([[robot_x, robot_y, robot_phi]])
    else:
        position_array[-1,:] = np.array([robot_x,robot_y,robot_phi])
    skirt_r = 70   # Sensor skirt radius
    obs_theta = math.pi/6
    goalX = np.array([600, 400])    # goal position

    data = {"screen":screen, "goalX":goalX, "vmax":0.5, "gtg_scaling":0.0001, "K_p":0.01, "ao_scaling":0.00005}

    # Create obstacles
    [radius, circ_x, circ_y] = create_circular_obsts(num_circ_obsts)

    obstacles = np.array([radius, circ_x, circ_y]).T

    # PyGame loop
    while(1):
        # To exit
        event = pygame.event.poll()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break
        screen.fill((50, 55, 60))   # background

        # Draw robot, sensor skirt, obstacles and goal
        bot = robot(robot_x, robot_y, robot_phi, robot_l, robot_b, data)
        # pygame.draw.circle(screen, (100, 100, 100), (int(bot.x), int(bot.y)), skirt_r, 0)   # Draw sensor skirt
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
                if random.random() <= 1:
                    close_radius.append(radius[i])
                    close_circ_x.append(circ_x[i])
                    close_circ_y.append(circ_y[i])
                    close_obst.append([circ_x[i], circ_y[i], radius[i]])
                    dist.append(distance)
        # Go to goal
        if(len(close_obst) == 0) or (math.sqrt((goalX[0] - robot_x)**2 + (goalX[1] - robot_y)**2) <= min(dist)):           # No obstacle in sensor skirt
            draw_circular_obsts(len(close_obst), close_radius, close_circ_x, close_circ_y, (255, 0, 0))
            # [v, omega] = bot.go_to_goal()   # output from controller go_to_goal()
            obs = torch.tensor([[0.0, 0.0, 0.0, 0.0]]).float()
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
            distance = math.sqrt((circ_x[i] - robot_x)**2 + (circ_y[i] - robot_y)**2)
            obstX = np.array([obs_x, obs_y])
            # [v, omega] = bot.avoid_obst(obstX, obs_radius)
            obs = torch.tensor([[1.0, obs_x/600, obs_y/600, distance/skirt_r]]).float()
        
        state = torch.tensor([[robot_x, robot_y, robot_phi]]).float()
        
        action = compute_action_diff(alphas_bar, alphas, betas, denoiser, state, obs)

        v = (action[0,0].item())*0.5
        omega = action[0,1].item()*0.02

        # print(v,omega)

        # if abs(v) > 0.5:
        #     v = 0.5
        
        # if abs(omega) > 0.02:
        #     omega = 0.02

        # Update robot position and orientation as per control input
        robot_x += v*math.cos(robot_phi); robot_y+= v*math.sin(robot_phi); robot_phi += omega; robot_phi = (robot_phi + np.pi) % (2 * np.pi) - np.pi
        position_array = np.append(position_array, np.array([[robot_x,robot_y,robot_phi]]),axis = 0)

        # FPS. Print if required
        clock.tick(3000)     # To limit fps, controls speed of the animation
        fps = (frames*1000)/(pygame.time.get_ticks() - ticks)   # calculate current fps

        # Update PyGame display
        pygame.display.flip()
        frames+=1

        distance_to_goal = math.sqrt((goalX[0] - robot_x)**2 + (goalX[1] - robot_y)**2)
        threshold = 15
        if distance_to_goal < threshold or frames - old_frame> 5000:
            # if run%20 == 0:
            #     print("Trajectory number:",run+1)
            old_frame = frames
            run += 1
            # np.savetxt("data/positions_"+str(n)+".csv", position_array, delimiter=",",fmt ='% s')
            # np.savetxt("data/obstacles_"+str(n)+".csv", obstacles, delimiter=",",fmt ='% s')
            break

if(__name__ == '__main__'):
    main()