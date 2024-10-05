#!/home/kang/anaconda3/envs/racecar/bin/python
import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import TransformStamped
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from diffusers.optimization import get_scheduler
from diffusers import DDPMScheduler, UNet2DConditionModel, DiffusionPipeline, DDIMScheduler
from diffusers.training_utils import EMAModel
import glob
import os
from tqdm.auto import tqdm
# import pygame
# from pygame.locals import *
# from unicycle_robot import *
from bicycle_robot import *
import random, time

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")

# Define the paths to the data directories
# data_directories = [
#     './data/monroe_data/1/480/2024-03-05T165359/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/480/2024-03-05T165336/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/480/2024-03-05T165313/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/480/2024-03-05T165247/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/480/2024-03-05T165230/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/96/2024-03-05T164823/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/96/2024-03-05T164850/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/96/2024-03-05T164912/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/96/2024-03-05T164934/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/96/2024-03-05T164956/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/192/2024-02-27T170725/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/192/2024-03-05T165017/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/192/2024-03-05T165045/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/192/2024-03-05T165102/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/192/2024-03-05T165126/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/192/2024-03-05T165143/',
#     './src/racecar/scripts/Remote/data/monroe_data/1/192/2024-03-05T165204/',
#     # './data/shivani_data/1/96/2024-03-27T234645/',
#     # './data/shivani_data/1/96/2024-03-27T234715/',
#     # './data/shivani_data/1/96/2024-03-27T234742/',
#     # './data/shivani_data/1/96/2024-03-27T234804/',
#     # './data/shivani_data/1/96/2024-03-27T234824/',
#     # './data/shivani_data/1/192/2024-03-27T234858/',
#     # './data/shivani_data/1/192/2024-03-27T234918/',
#     # './data/shivani_data/1/192/2024-03-27T234939/',
#     # './data/shivani_data/1/192/2024-03-27T234957/',
#     # './data/shivani_data/1/192/2024-03-27T235023/',
#     # './data/shivani_data/1/480/2024-03-27T235037/',
#     # './data/shivani_data/1/480/2024-03-27T235054/',
#     # './data/shivani_data/1/480/2024-03-27T235111/',
#     # './data/shivani_data/1/480/2024-03-27T235128/',
#     # './data/shivani_data/1/480/2024-03-27T235145/',
# ]


# Function to load data from a directory
def load_data_from_directory(data_directory):
    trajectory_file = data_directory + 'trajectory_data.json'
    with open(trajectory_file) as f:
        trajectory_data = json.load(f)
    
    # Convert theta from degrees to radians
    for step in trajectory_data:
        step['theta'] = np.radians(step['theta'])  # Convert to radians
    
    obstacle_files = sorted(glob.glob(data_directory + 'obstacle*_trajectory_data.json'))
    obstacle_data = []
    for file in obstacle_files:
        with open(file) as f:
            obstacle_data.append(json.load(f))
    
    
    return trajectory_data, obstacle_data


# # Load all datasets
# all_trajectory_data = []
# all_obstacle_data = []

# for data_directory in data_directories:
#     trajectory_data, obstacle_data = load_data_from_directory(data_directory)
#     all_trajectory_data.append(trajectory_data)
#     all_obstacle_data.append(obstacle_data)

# print("Loaded all datasets.")
# print(f"Number of trajectory datasets: {len(all_trajectory_data)}")
# print(f"Number of obstacle datasets: {len(all_obstacle_data)}")

# # Use the first dataset for testing
# test_trajectory_data = all_trajectory_data[0]
# test_obstacle_data = all_obstacle_data[0]
#print(len(test_obstacle_data[0]))

# Parameters
view_range = 480  # Adjust to 192 or 480 as needed
view_angle = 45  # View angle in degrees
history_length = 100  # Number of historical steps to include

# Helper functions for coordinate transformations
def transform_to_local_frame(x, y, agent_x, agent_y, agent_theta):
    dx = x - agent_x
    dy = y - agent_y
    local_x = dx * np.cos(agent_theta) + dy * np.sin(agent_theta)
    local_y = -dx * np.sin(agent_theta) + dy * np.cos(agent_theta)
    return local_x, local_y

def generate_closest_obstacles(agent_x, agent_y, agent_theta, obstacle_data, view_range, view_angle, time_step):
    
    visible_obstacles = []
    for obstacle_trajectory in obstacle_data:
        if len(obstacle_trajectory) == 1:  # Stationary obstacle
            obstacle = obstacle_trajectory[0]
        else:  # Moving obstacle
            obstacle = obstacle_trajectory[min(time_step, len(obstacle_trajectory) - 1)]
        
        global_x, global_y = obstacle['x'], obstacle['y']
        distance = np.sqrt((global_x - agent_x)**2 + (global_y - agent_y)**2)
        relative_angle = np.arctan2(global_y - agent_y, global_x - agent_x) - agent_theta
        relative_angle = np.degrees(relative_angle)
        if relative_angle > 180:
            relative_angle -= 360
        elif relative_angle < -180:
            relative_angle += 360
        if distance <= view_range and -view_angle/2 <= relative_angle <= view_angle/2:
            visible_obstacles.append((global_x, global_y, distance))
    
    visible_obstacles.sort(key=lambda x: x[2])
    closest_obstacles = visible_obstacles[:3]
    
    if len(closest_obstacles) == 0:
        closest_obstacles_array = np.array([[0, 0], [0, 0], [0, 0]])
    else:
        closest_obstacles_array = np.array([list(obstacle[:2]) for obstacle in closest_obstacles])
        while closest_obstacles_array.shape[0] < 3:
            closest_obstacles_array = np.vstack([closest_obstacles_array, [0, 0]])
    
    return closest_obstacles_array.flatten()

def generate_state_with_closest_obstacles(agent_trajectory, obstacle_data, goal, history_length, view_range, view_angle):
    inputs = []
    actions = []
    empty_history = [0, 0, 0] * history_length

    for i in range(len(agent_trajectory)):
        agent_x, agent_y, agent_theta = agent_trajectory[i]['x'], agent_trajectory[i]['y'], agent_trajectory[i]['theta']
        if i < len(agent_trajectory) - 1:
            next_agent_x, next_agent_y, next_agent_theta = agent_trajectory[i+1]['x'], agent_trajectory[i+1]['y'], agent_trajectory[i+1]['theta']
        else:
            next_agent_x, next_agent_y, next_agent_theta = agent_x, agent_y, agent_theta
        delta_x = next_agent_x - agent_x
        delta_y = next_agent_y - agent_y
        delta_theta = next_agent_theta - agent_theta
        closest_obstacles = generate_closest_obstacles(agent_x, agent_y, agent_theta, obstacle_data, view_range, view_angle, i)
        combined_input = np.concatenate(([agent_x, agent_y, agent_theta], closest_obstacles))
        inputs.append(combined_input)
        actions.append([delta_x, delta_y, delta_theta])

    return np.array(inputs), np.array(actions)

# # Generate training data with obstacle and goal information
# train_inputs = []
# train_actions = []

# goal = (all_trajectory_data[0][-1]['x'], all_trajectory_data[0][-1]['y'])

# for i in range(len(all_trajectory_data) - 1):
#     inputs, actions = generate_state_with_closest_obstacles(
#         all_trajectory_data[i], all_obstacle_data[i], goal, history_length, view_range, view_angle)

#     train_inputs.append(inputs)
#     train_actions.append(actions)

# # Convert lists to arrays
# train_inputs = np.vstack(train_inputs)
# train_actions = np.vstack(train_actions)

# print("Before Normalization:", train_inputs[100])

max_x = 1204.705078125
max_y = 677.378173828125   

print('max_x', max_x)
print('max_y', max_y)

# train_inputs[:, 0] /= max_x
# train_inputs[:, 1] /= max_y
# train_inputs[:, 2] /= np.pi


# Normalize obstacle coordinates
# Assuming obstacle coordinates start from column 3 onwards and are interleaved as [obs_x1, obs_y1, obs_x2, obs_y2, ...]
# obstacle_x_columns = train_inputs[:, 3::2]  # Every second column starting from 3 corresponds to obstacle x coordinates
# obstacle_y_columns = train_inputs[:, 4::2]  # Every second column starting from 4 corresponds to obstacle y coordinates

max_obstacle_x = 1150
max_obstacle_y = 640

# # Normalize the obstacle coordinates
# train_inputs[:, 3::2] /= max_obstacle_x  # Normalize obstacle x coordinates
# train_inputs[:, 4::2] /= max_obstacle_y  # Normalize obstacle y coordinates

max_delta_x = 4.896728515625  
max_delta_y = 4.322685241699197 
max_delta_theta = np.pi

print('max_delta_x', max_delta_x)
# print(max_delta_y)
# print(np.mean(train_actions[:, 0]))
# print(np.mean(train_actions[:, 1]))
# print(np.min(np.abs(train_actions[:, 0])))
# print(np.min(np.abs(train_actions[:, 1])))
print('max_obstacle_x', max_obstacle_x)
print('max_obstacle_y', max_obstacle_y)
# train_actions[:, 0] /= max_delta_x
# train_actions[:, 1] /= max_delta_y
# train_actions[:, 2] /= max_delta_theta

# print("After Normalization:", train_inputs[100])

def normalize_obstacles(obstacle_data, max_obstacle_x, max_obstacle_y):
    normalized_obstacles = []

    for obstacle_trajectory in obstacle_data:
        normalized_trajectory = []
        for step in obstacle_trajectory:
            normalized_x = step['x'] / max_obstacle_x
            normalized_y = step['y'] / max_obstacle_y
            normalized_trajectory.append({'x': normalized_x, 'y': normalized_y})
        normalized_obstacles.append(normalized_trajectory)

    return normalized_obstacles

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()  # Final activation can be included or omitted depending on the use case
        )

    def forward(self, x):
        return self.network(x)

class ClassConditionedUnet(nn.Module):
    def __init__(self, location_emb_size=1280, hidden_dim=512, output_size=16):  # `output_size` is the desired spatial size (e.g., 16x16)
        super().__init__()

        # Linear layer to map (x, y, theta) to a vector of size location_emb_size
        #self.location_emb = nn.Linear(9, location_emb_size)  # Adjust input size based on your context (e.g., location and obstacles)
        self.location_emb = MLP(input_dim=9, hidden_dim=hidden_dim, output_dim=location_emb_size)  # Adjust input size based on your context (e.g., location and obstacles)

        # MLP to project noisy_action to a higher spatial dimension
        self.project_to_high_dim = MLP(3, hidden_dim, 3 * output_size * output_size)  # Input 3, hidden_dim intermediate, output 3 * H * W
        self.reduce_to_output = nn.Linear(3 * output_size * output_size, 3)

        # Define the conditional UNet model with cross-attention
        self.model = UNet2DConditionModel(
            sample_size=output_size,  # The input image size (e.g., 16x16 after projection)
            in_channels=3,  # The number of input channels (noisy action)
            out_channels=3,  # The number of output channels (predicted noise)
            layers_per_block=2,
            #block_out_channels=(64, 128, 256),
            block_out_channels=(64, 128, 256, 512),
            # down_block_types=(
            #     "CrossAttnDownBlock2D",  # Use cross-attention in the downsampling path
            #     "CrossAttnDownBlock2D",
            #     "CrossAttnDownBlock2D",
            # ),
            # up_block_types=(
            #     "CrossAttnUpBlock2D",  # Use cross-attention in the upsampling path
            #     "CrossAttnUpBlock2D",  
            #     "CrossAttnUpBlock2D",  # You can keep the final block as standard or use cross-attention here as well
            # ),
            # mid_block_type="UNetMidBlock2DCrossAttn",  # Use cross-attention in the middle block
            mid_block_type="UNetMidBlock2DCrossAttn",  # Use cross-attention in the middle block
            cross_attention_dim=location_emb_size,  # The dimension of the cross-attention
        )

    def forward(self, noisy_action, t, location):
        # Project noisy_action to higher spatial dimensions using the MLP
        output_size = 16
        projected_action = self.project_to_high_dim(noisy_action)  # Shape: [batch_size, 3 * output_size * output_size]
        projected_action = projected_action.view(noisy_action.size(0), 3, output_size, output_size)  # Reshape to [batch_size, 3, output_size, output_size]

        # Embed the current location and obstacles (context)
        location_emb = self.location_emb(location)  # Shape: [batch_size, location_emb_size]
        
        # Pass through the U-Net model with cross-attention conditioning
        location_emb = location_emb.unsqueeze(1)  # Add a dimension for broadcasting
        output = self.model(projected_action, timestep=t, encoder_hidden_states=location_emb).sample  # Output shape: [batch_size, 3, output_size, output_size]

        # Flatten and pass through a fully connected layer
        output = output.view(output.size(0), -1)  # Flatten: [batch_size, 3 * output_size * output_size]
        output = self.reduce_to_output(output)  # Final projection to [batch_size, 3]
        
        return output

# Initialize the model
model = ClassConditionedUnet(location_emb_size=9).to(device)

print("Model initialized.")

# Filepath for saving/loading the model
model_path = '/src/racecar/scripts/Remote/checkpoints/best_diffusion_model_with_obstacles_2.pth'

noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=100,
    beta_schedule="squaredcos_cap_v2")

ddim_scheduler.set_timesteps(10)

print("Model and scheduler initialized.")

ema = EMAModel(
        parameters=model.parameters(),
        power=0.75
    )

print("Loading pre-trained model...")
# Load the pre-trained model
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])  # Load EMA state
    print(f"Model loaded from {model_path}")
else:
    print(f"No pre-trained model found at {model_path}. Please set `train_model = True` to train a new model.")
    
# Screen
screen_width = 420; screen_height = 360
# screen = pygame.display.set_mode([1000, 800], DOUBLEBUF)

# Obstacles
num_circ_obsts = 7; obst_min_radius = 10; obst_max_radius = 10  # for circular obstacles

def create_circular_obsts(num):
    radius = []; circ_x = []; circ_y = []
    for i in range(num):
        radius.append(random.randint(obst_min_radius, obst_max_radius))
        circ_x.append(random.randint(radius[i], screen_width - radius[i]))
        circ_y.append(random.randint(radius[i], screen_height - radius[i]))
    return [radius, circ_x, circ_y]

# def draw_circular_obsts(num, radius, circ_x, circ_y, color):
#     for i in range(num):
#         pygame.draw.circle(screen, color, (circ_x[i], circ_y[i]), radius[i], 0)
        
def angle_truth_value(robot_x, robot_y, robot_phi, obs_theta, circ_x, circ_y, r, distance_to_obstacle):
    relative_angle = math.atan2(circ_y - robot_y, circ_x - robot_x)
    truth_value = False
    if abs(relative_angle - robot_phi) <= obs_theta:
        truth_value = True
    elif abs(relative_angle - robot_phi ) <= obs_theta + math.atan2(r, distance_to_obstacle):
        truth_value = True
    return truth_value

def create_obsts_fixed(x_positions, y_positions):
    radius = []; circ_x = []; circ_y = []
    print(len(x_positions))
    for i in range(len(x_positions)):
        radius.append(random.randint(obst_min_radius, obst_max_radius))
        circ_x.append((x_positions[i])*420/1000)
        circ_y.append((y_positions[i])*360/600)
    return [radius, circ_x, circ_y]

def create_circular_obsts2(num, original_obstacles, lab_width, lab_height, screen_width, screen_height, obst_min_radius, obst_max_radius):
    radius = []
    circ_x = []
    circ_y = []
    
    # Calculate scaling factors for x and y
    scale_x = screen_width / lab_width
    scale_y = screen_height / lab_height
    
    # Scale the original obstacles' coordinates
    scaled_obstacles = []
    for obs in original_obstacles:
        scaled_x = obs[0] * scale_x
        scaled_y = obs[1] * scale_y
        scaled_obstacles.append([scaled_x, scaled_y])
    
    # Add scaled obstacles to the list
    for i in range(len(scaled_obstacles)):
        radius.append(random.randint(obst_min_radius, obst_max_radius))  # Random radius
        circ_x.append(int(scaled_obstacles[i][0]))  # Scaled and rounded x coordinate
        circ_y.append(int(scaled_obstacles[i][1]))  # Scaled and rounded y coordinate
    
    return [radius, circ_x, circ_y]


# Global variables for storing pose data
current_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}

# Create ROS publishers
spub = rospy.Publisher('steer', Float32, queue_size=1)
tpub = rospy.Publisher('throttle', Float32, queue_size=1)

# Initialize the ROS node
rospy.init_node('G29_controller', anonymous=True, disable_signals=True)

def clamp(vmin, val, vmax):
    return min(max(vmin, val), vmax)

def vicon_callback(msg):
    global current_pose
    # Extract pose information from TransformStamped
    current_pose['x'] = msg.transform.translation.x
    current_pose['y'] = msg.transform.translation.y
    # current_pose['z'] = msg.transform.translation.z
    # Convert orientation quaternion to Euler angles
    orientation_q = msg.transform.rotation
    roll, pitch, yaw = euler_from_quaternion(
        orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
    )
    # current_pose['roll'] = roll
    # current_pose['pitch'] = pitch
    current_pose['yaw'] = yaw

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(clamp(-1, sinp, 1))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp) + math.pi/2
    
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

    return roll, pitch, yaw

def compute_steer_command(current_pose, desired_pose):
    error_x = desired_pose['x'] - current_pose['x']
    steer = 128 + error_x * 10  # Example computation
    steer = 255
    return steer

def compute_throttle_command(current_pose, desired_pose):
    error_z = desired_pose['x'] - current_pose['x']
    throttle = 0 + error_z * 100  # Example computation
    throttle = 0.0
    return throttle

# Create a subscriber for Vicon data
rospy.Subscriber('/vicon/Racecar/Racecar', TransformStamped, vicon_callback)

rate = rospy.Rate(10)  # 10 Hz

try:
    robot_l = 38; robot_b = 19 
    robot_x = current_pose['x']
    robot_y = current_pose['y']
    robot_theta = current_pose['yaw']
    robot_phi = 0
    robot_obs = 0
    position_array = np.array([[robot_x, robot_y, robot_theta, robot_phi]])
    skirt_r = 480*420/1100   # Sensor skirt radius
    obs_theta = math.pi/6
    goalX = np.array([1050 * 420 / 1100, 620 * 360 / 650])    # goal position
    data = {"goalX":goalX, "vmax":10, "gtg_scaling":0.0001, "K_p":0.5, "ao_scaling":0.00005}
    # Create obstacles
    ######
    original_obstacles = [
        #[754.671, 412.418],
        [344.486, 508.149],
        #[127.251, 240.864],
        #[535.735, 120.297],
        #[754.413, 204.740],
        #[973.505, 319.264],
        [483.842, 299.161],
        #[752.892, 412.535],
        [653.537, 124.185],
        #[30.911, 65.518],
        [498.017, 201.182],
        [601.642, 118.530],
        [871.514, 362.297],
        [538.469, 270.722],
        [505.249, 478.287],
        [606.248, 403.393],
        [367.165, 306.231],
        #[871.000, 303.273],
        #[377.034, 197.638],
    ]

    # Lab and screen dimensions
    lab_width = 1100
    lab_height = 650
    screen_width = 420
    screen_height = 360
    obst_min_radius = 10
    obst_max_radius = 30

    # Create circular obstacles with scaling
    # num_obstacles = 10
    #[radius, circ_x, circ_y] = create_circular_obsts2(num_obstacles, original_obstacles, lab_width, lab_height, screen_width, screen_height, obst_min_radius, obst_max_radius)
    ######
    # [radius, circ_x, circ_y] = create_circular_obsts(num_circ_obsts)

    # obstacles = np.array([radius, circ_x, circ_y]).T
    
    # x_positions = [754.671, 344.486, 127.251, 535.735, 754.413, 973.505, 483.842, 752.892, 653.537, 30.911, 498.017, 601.642, 871.514, 538.469, 505.249, 606.248, 367.165, 871.000, 377.034, 452.582]
    # y_positions = [412.418, 508.149, 240.864, 120.297, 204.740, 319.264, 299.161, 412.535, 124.185, 65.518, 201.182, 118.530, 362.297, 270.722, 478.287, 403.393, 306.231, 303.273, 197.638, 211.339]
    # [radius, circ_x, circ_y] = create_obsts_fixed(x_positions, y_positions)
    
    x_positions = [344.486, 483.842, 601.642, 871.514, 505.249, 606.248, 367.165]
    y_positions = [508.149, 299.161, 118.530, 362.297, 478.287, 403.393, 306.231]
    [radius, circ_x, circ_y] = create_obsts_fixed(x_positions, y_positions)

    throttle = 0
    steer = 128
    obstacles = np.array([radius, circ_x, circ_y]).T
    
    frame = 0
    while not rospy.is_shutdown():
        
        position_array = np.append(position_array, np.array([[robot_x,robot_y,robot_theta, robot_phi, robot_obs]]),axis = 0)
        
        bot = BicycleRobot(robot_x, robot_y, robot_theta, robot_phi, robot_obs, robot_l, robot_b, data)
        
        close_obst = []
        dist = []
        close_radius = []
        close_circ_x = []
        close_circ_y = []
        for i in range(num_circ_obsts):
            distance = math.sqrt((circ_x[i] - robot_x)**2 + (circ_y[i] - robot_y)**2)
            if (distance <= (skirt_r + radius[i]) and angle_truth_value(robot_x, robot_y, robot_obs, obs_theta, circ_x[i], circ_y[i], radius[i], distance)):
            # if True:
                if random.random() <= 1:
                    close_radius.append(radius[i])
                    close_circ_x.append(circ_x[i])
                    close_circ_y.append(circ_y[i])
                    # Append distance instead of radius
                    close_obst.append([circ_x[i], circ_y[i], distance])
                    dist.append(distance)
        
        # Sort obstacles by distance (third element)
        close_obst.sort(key=lambda x: x[2])

        # Select the 3 closest obstacles
        closest_obstacles = close_obst[:3]
        if len(closest_obstacles) == 0:
            closest_obstacles_array = np.array([[0, 0], [0, 0], [0, 0]])
        else:
            closest_obstacles_array = np.array([list(obstacle[:2]) for obstacle in closest_obstacles])
            while closest_obstacles_array.shape[0] < 3:
                closest_obstacles_array = np.vstack([closest_obstacles_array, [0, 0]])

        closest_obstacles = np.array(closest_obstacles_array.flatten(), dtype=np.float32)
        
        closest_obstacles[0::2] = closest_obstacles[0::2]*lab_width/420
        closest_obstacles[1::2] = closest_obstacles[1::2]*lab_height/360
        
        # Normalize obstacles
        closest_obstacles[0::2] /= max_obstacle_x  # Normalize x-coordinates of obstacles
        closest_obstacles[1::2] /= max_obstacle_y  # Normalize y-coordinates of obstacles

        # Concatenate the agent's state with normalized obstacles
        state_with_obstacles = np.concatenate(([(robot_x*lab_width/420)/max_x, (robot_y*lab_height/360)/max_y, robot_obs/np.pi], closest_obstacles))
        state_tensor = torch.tensor(state_with_obstacles, dtype=torch.float32).unsqueeze(0).to(device)

        # Generate a noisy action (with batch dimension)
        random_action = torch.randn(1, 3).to(device)

        # Expand random_action to match the UNet output shape [batch_size, 3, 32, 32]
        #random_action_expanded = random_action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 32)

        #random_action_expanded = random_action.view(random_action.size(0), random_action.size(1), 1, 1)
        #print(random_action_expanded.shape)

        random_action_expanded = random_action
        
        # using ddim scheduler to denosier with 20 steps of gap
        for _, step in enumerate(ddim_scheduler.timesteps):
            # if step%6 == 0:
                # tpub.publish(Float32(float(clamp(-255, 0.0, 255))))
            t_step = torch.tensor([step], dtype=torch.long).to(device)
            noisy_inputs = random_action_expanded  # Use the expanded version for input
            #print(noisy_inputs.shape)
            noise_pred = model(noisy_inputs, t_step, state_tensor)
            result = ddim_scheduler.step(model_output=noise_pred, timestep=t_step, sample=random_action_expanded)
            random_action_expanded = result.prev_sample  # Extract the previous timestep sample
            
        # tpub.publish(Float32(float(clamp(-255, 0.0, 255))))
        
        for _, step in enumerate(noise_scheduler.timesteps[-10:]):
            if step%7 == 0:
                tpub.publish(Float32(float(clamp(-255, 0.0, 255))))
            #print('11111111111111111111111111111111')
            t_step = torch.tensor([step], dtype=torch.long).to(device)
            noisy_inputs = random_action_expanded  # Use the expanded version for input
            #print(noisy_inputs.shape)
            noise_pred = model(noisy_inputs, t_step, state_tensor)
            result = noise_scheduler.step(model_output=noise_pred, timestep=t_step, sample=random_action_expanded)
            random_action_expanded = result.prev_sample  # Extract the previous timestep sample
            #random_action_expanded = random_action_expanded.mean(dim=[2, 3], keepdim=True)

        # After denoising, take the mean over spatial dimensions to reduce back to shape [1, 3]
        action_pred = random_action_expanded.squeeze().cpu()
        delta_x, delta_y, delta_phi = action_pred.detach().numpy()

        # Update robot position and orientation as per control input
        
        de_robot_x = robot_x * lab_width / 420
        de_robot_y = robot_y * lab_height / 360
        de_robot_phi = robot_obs

        a_robot_x = delta_x * max_delta_x
        a_robot_y = delta_y * max_delta_y
        a_robot_phi = delta_phi * max_delta_theta

        de_robot_x += a_robot_x * 50
        de_robot_y += a_robot_y * 50
        de_robot_phi += a_robot_phi * 50

        if de_robot_phi > np.pi:
            de_robot_phi -= 2 * np.pi
        elif de_robot_phi < -np.pi:
            de_robot_phi += 2 * np.pi

        robot_x_new = de_robot_x * 420 / lab_width
        robot_y_new = de_robot_y * 360 / lab_height
        robot_phi_new = de_robot_phi

        delta_x = robot_x_new - robot_x
        delta_y = robot_y_new - robot_y
        delta_phi = robot_phi_new - robot_obs
        
        
        v, phi_rate = bot.compute_controls(delta_x, delta_y)
        
        robot_phi = (steer - 128)*2*np.pi/(3*255)
        
        robot_phi_d = robot_phi + phi_rate
        
        dt = 1
        
        print("Phi desired:",robot_phi_d)
        
        # robot_phi = robot_phi + phi_rate*dt
        # robot_phi = (robot_phi + np.pi) % (2 * np.pi)
        
        # print("robot phi:",robot_phi)
        
        # Compute control commands based on pose feedback
        steer = -255*robot_phi_d/(2*np.pi/3) + 128
        throttle = v * 225 / 10 
        
        if steer < 128:
            steer -= 20
        
        if frame == 0 or frame == 1:
            steer = 101
            throttle = 255
        
        # Publish data to ROS topics
        spub.publish(Float32(float(clamp(0, steer, 255))))
        # time.sleep(0.1)
        tpub.publish(Float32(float(clamp(-255, throttle, 255))))
        # time.sleep(1)
        
        robot_x = current_pose['x']*100 + 180
        robot_y = current_pose['y']*100 + 180
        robot_theta = current_pose['yaw']
        robot_phi = (steer - 128)*2*np.pi/(3*255)
        
        robot_obs += delta_phi
        robot_obs = (robot_obs + np.pi) % (2 * np.pi) - np.pi
        
        frame = frame + 1

        # Print for debugging
        rospy.loginfo(f'Published steer: {steer}, throttle: {throttle}')
        print("Current Pose: ", current_pose)
        
except KeyboardInterrupt:
    #print current directory
    print(os.getcwd())
    np.savetxt("../Documents/maulikbhatt/Racecar/src/racecar/scripts/Remote/bicycle_diff_data/bicycle_full_positions_expt"+".csv", position_array, delimiter=",",fmt ='% s')
    rospy.loginfo("Ctrl+C detected, stopping...")
    # Publish zero values before shutting down
    spub.publish(Float32(128.0))
    tpub.publish(Float32(0.0))
    rospy.signal_shutdown("Keyboard interrupt")