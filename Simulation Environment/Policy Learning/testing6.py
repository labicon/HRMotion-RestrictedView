import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge
from IPython.display import HTML
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold
import math
from torch.cuda.amp import GradScaler, autocast

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the paths to the data directories
data_directories = [
    './data/monroe_data/1/480/2024-03-05T165359/',
    './data/monroe_data/1/480/2024-03-05T165336/',
    './data/monroe_data/1/480/2024-03-05T165313/',
    './data/monroe_data/1/480/2024-03-05T165247/',
    './data/monroe_data/1/480/2024-03-05T165230/',
    './data/monroe_data/1/96/2024-03-05T164823/',
    './data/monroe_data/1/96/2024-03-05T164850/',
    './data/monroe_data/1/96/2024-03-05T164912/',
    './data/monroe_data/1/96/2024-03-05T164934/',
    './data/monroe_data/1/96/2024-03-05T164956/'
]

# Function to load data from a directory
def load_data_from_directory(data_directory):
    trajectory_file = data_directory + 'trajectory_data.json'
    with open(trajectory_file) as f:
        trajectory_data = json.load(f)
    
    obstacle_files = sorted(glob.glob(data_directory + 'obstacle*_trajectory_data.json'))
    obstacle_data = []
    for file in obstacle_files:
        with open(file) as f:
            obstacle_data.append(json.load(f))
    
    return trajectory_data, obstacle_data

# Load all datasets
all_trajectory_data = []
all_obstacle_data = []

for data_directory in data_directories:
    trajectory_data, obstacle_data = load_data_from_directory(data_directory)
    all_trajectory_data.append(trajectory_data)
    all_obstacle_data.append(obstacle_data)

print("Loaded all datasets.")
print(f"Number of trajectory datasets: {len(all_trajectory_data)}")
print(f"Number of obstacle datasets: {len(all_obstacle_data)}")

# Plot all agent trajectories for comparison
plt.figure(figsize=(12, 8))
for trajectory_data in all_trajectory_data:
    xs = [point['x'] for point in trajectory_data]
    ys = [point['y'] for point in trajectory_data]
    plt.plot(xs, ys, marker='o')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('All Agent Trajectories')
plt.grid(True)
plt.show()

# Combine all data except one for testing
train_trajectory_data = []
train_obstacle_data = []

# Use the last dataset for testing
test_trajectory_data = all_trajectory_data[-1]
test_obstacle_data = all_obstacle_data[-1]

# Combine the rest for training
for i in range(len(all_trajectory_data) - 1):
    train_trajectory_data.extend(all_trajectory_data[i])
    train_obstacle_data.extend(all_obstacle_data[i])

def transform_to_local_frame(x, y, agent_x, agent_y, agent_theta):
    dx = x - agent_x
    dy = y - agent_y
    local_x = dx * np.cos(agent_theta) + dy * np.sin(agent_theta)
    local_y = -dx * np.sin(agent_theta) + dy * np.cos(agent_theta)
    return local_x, local_y

def transform_to_global_frame(local_x, local_y, agent_x, agent_y, agent_theta):
    global_x = local_x * np.cos(agent_theta) - local_y * np.sin(agent_theta) + agent_x
    global_y = local_x * np.sin(agent_theta) + local_y * np.cos(agent_theta) + agent_y
    return global_x, global_y

def is_within_view(local_x, local_y, view_range, view_angle):
    distance = np.sqrt(local_x**2 + local_y**2)
    angle = np.arctan2(local_y, local_x) * 180 / np.pi
    if distance <= view_range and -view_angle/2 <= angle <= view_angle/2:
        return True
    return False

# Parameters
view_range = 480  # Change to 192 or 480 as needed
view_angle = 45   # View angle in degrees
grid_size = 4     # Increase grid size to reduce memory usage

grid_width = int(view_range * 2 / grid_size)
grid_height = int(view_range * 2 / grid_size)

# Function to generate occupancy maps and actions with obstacle information
def generate_occupancy_maps_and_actions_with_obstacles(agent_trajectory, obstacle_data):
    occupancy_maps = []
    actions = []

    for i in range(1, len(agent_trajectory)):
        agent_x, agent_y, agent_theta = agent_trajectory[i-1]['x'], agent_trajectory[i-1]['y'], agent_trajectory[i-1]['theta']
        next_agent_x, next_agent_y, next_agent_theta = agent_trajectory[i]['x'], agent_trajectory[i]['y'], agent_trajectory[i]['theta']
        
        delta_x = next_agent_x - agent_x
        delta_y = next_agent_y - agent_y
        delta_theta = next_agent_theta - agent_theta
        
        occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.float32)
        
        for obstacle_trajectory in obstacle_data:
            if len(obstacle_trajectory) == 1:  # Stationary obstacle
                obstacle = obstacle_trajectory[0]
            else:  # Moving obstacle
                obstacle = obstacle_trajectory[min(i, len(obstacle_trajectory) - 1)]
                
            local_x, local_y = transform_to_local_frame(obstacle['x'], obstacle['y'], agent_x, agent_y, agent_theta)
            
            grid_x = int((local_x + view_range) / grid_size)
            grid_y = int((local_y + view_range) / grid_size)
            
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                if is_within_view(local_x, local_y, view_range, view_angle):
                    occupancy_grid[grid_y, grid_x] = 1  # Visible obstacle
                else:
                    if occupancy_grid[grid_y, grid_x] == 0:  # Avoid overwriting if already marked as visible
                        occupancy_grid[grid_y, grid_x] = 2  # Invisible obstacle
        
        state_with_obstacles = np.concatenate([occupancy_grid.flatten(), [agent_x, agent_y, agent_theta]], axis=0)
        occupancy_maps.append(state_with_obstacles)
        actions.append([delta_x, delta_y, delta_theta])

    return np.array(occupancy_maps), np.array(actions)

# Generate training data with obstacle information
train_occupancy_maps = []
train_actions = []

for i in range(len(all_trajectory_data) - 1):
    occupancy_maps, actions = generate_occupancy_maps_and_actions_with_obstacles(all_trajectory_data[i], all_obstacle_data[i])
    train_occupancy_maps.append(occupancy_maps)
    train_actions.append(actions)

# Convert lists to arrays
train_occupancy_maps = np.concatenate(train_occupancy_maps, axis=0).astype(np.float32)
train_actions = np.concatenate(train_actions, axis=0).astype(np.float32)

print(f"Train occupancy maps shape: {train_occupancy_maps.shape}")
print(f"Train actions shape: {train_actions.shape}")

# Split data into training and validation sets
train_occupancy_maps, val_occupancy_maps, train_actions, val_actions = train_test_split(train_occupancy_maps, train_actions, test_size=0.2, random_state=42)

print(f"Train occupancy maps shape after split: {train_occupancy_maps.shape}")
print(f"Validation occupancy maps shape: {val_occupancy_maps.shape}")
print(f"Train actions shape after split: {train_actions.shape}")
print(f"Validation actions shape: {val_actions.shape}")

# Downsample the occupancy maps
def downsample(grid, factor=4):
    reshaped = grid.reshape((grid.shape[0] // factor, factor, grid.shape[1] // factor, factor))
    downsampled = reshaped.mean(axis=(1, 3))
    return downsampled

# Ensure the maps have the correct shape before downsampling
expected_grid_height = grid_height
expected_grid_width = grid_width

train_occupancy_maps = train_occupancy_maps[:, :-3].reshape((train_occupancy_maps.shape[0], expected_grid_height, expected_grid_width))
val_occupancy_maps = val_occupancy_maps[:, :-3].reshape((val_occupancy_maps.shape[0], expected_grid_height, expected_grid_width))

train_occupancy_maps = np.array([downsample(grid) for grid in train_occupancy_maps])
val_occupancy_maps = np.array([downsample(grid) for grid in val_occupancy_maps])

# Flatten the downsampled occupancy maps for the model input
train_occupancy_maps = train_occupancy_maps.reshape(train_occupancy_maps.shape[0], -1)
val_occupancy_maps = val_occupancy_maps.reshape(val_occupancy_maps.shape[0], -1)

# Debug prints to check shapes
print(f"Downsampled train occupancy maps shape: {train_occupancy_maps.shape}")
print(f"Downsampled val occupancy maps shape: {val_occupancy_maps.shape}")

# Update model input size
input_size = train_occupancy_maps.shape[1]
print(f"Model input size: {input_size}")

# Define the enhanced denoising model
class EnhancedDenoisingModel(nn.Module):
    def __init__(self, input_size):
        super(EnhancedDenoisingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 3)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model = EnhancedDenoisingModel(input_size).to(device)
print(model)

# Normalize actions
max_delta_x = np.max(np.abs(train_actions[:, 0]))
max_delta_y = np.max(np.abs(train_actions[:, 1]))
max_delta_theta = np.pi

train_actions[:, 0] /= max_delta_x  # Normalize delta x
train_actions[:, 1] /= max_delta_y  # Normalize delta y
train_actions[:, 2] /= max_delta_theta  # Normalize delta theta to be within [-1, 1]

# Define the diffusion process
class DiffusionProcess:
    def __init__(self, num_steps, beta_start=0.1, beta_end=0.2):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_t = self.alpha_cumprod[t].view(-1, 1)  # Reshape alpha_t to match the dimensions of x0
        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        return xt, noise

    def denoise(self, xt, t, noise_pred):
        alpha_t = self.alpha_cumprod[t].view(-1, 1)  # Reshape alpha_t to match the dimensions of xt
        noise_pred = noise_pred.view(-1, 1)  # Reshape noise_pred to match the dimensions of xt
        x0_pred = (xt - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        return x0_pred

diffusion = DiffusionProcess(num_steps=1000)

# Function to augment data
def augment_data(data, augmentation_factor=2):
    augmented_data = []
    for _ in range(augmentation_factor):
        for sample in data:
            noise = np.random.normal(0, 0.01, sample.shape)
            augmented_data.append(sample + noise)
    return np.array(augmented_data)

# Augment the training data
train_occupancy_maps = augment_data(train_occupancy_maps)
train_actions = augment_data(train_actions)

# Training the model with L2 regularization
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight_decay for L2 regularization

train_dataset = TensorDataset(torch.tensor(train_occupancy_maps, dtype=torch.float32).to(device), torch.tensor(train_actions, dtype=torch.float32).to(device))
val_dataset = TensorDataset(torch.tensor(val_occupancy_maps, dtype=torch.float32).to(device), torch.tensor(val_actions, dtype=torch.float32).to(device))

# Set a smaller batch size and accumulation steps
batch_size = 16
accumulation_steps = 4

num_epochs = 50  # Increase number of epochs
best_val_loss = float('inf')
early_stopping_patience = 10
patience_counter = 0

kf = KFold(n_splits=5)
scaler = GradScaler()

# Train only if the model is not already saved
if not torch.cuda.is_available() or not torch.cuda.device_count() > 0:
    for train_index, val_index in kf.split(train_occupancy_maps):
        train_loader = DataLoader(TensorDataset(torch.tensor(train_occupancy_maps[train_index], dtype=torch.float32).to(device), torch.tensor(train_actions[train_index], dtype=torch.float32).to(device)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(train_occupancy_maps[val_index], dtype=torch.float32).to(device), torch.tensor(val_actions[val_index], dtype=torch.float32).to(device)), batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()
            
            for i, (inputs, targets) in enumerate(train_loader):
                t = torch.randint(0, diffusion.num_steps, (inputs.size(0),)).to(device)
                noisy_inputs, noise = diffusion.add_noise(inputs, t)
                
                with autocast():
                    outputs = model(noisy_inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()

                # Perform optimization step after accumulating gradients
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * inputs.size(0)
            
            # Handle remaining gradients
            if (i + 1) % accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    t = torch.randint(0, diffusion.num_steps, (inputs.size(0),)).to(device)
                    noisy_inputs, noise = diffusion.add_noise(inputs, t)
                    
                    with autocast():
                        outputs = model(noisy_inputs)
                        loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_enhanced_diffusion_model_with_obstacles.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping")
                    break
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Load the best model
model.load_state_dict(torch.load('best_enhanced_diffusion_model_with_obstacles.pth'))

# Function to dynamically generate the occupancy map based on the current state
def generate_dynamic_occupancy_map(agent_x, agent_y, agent_theta, obstacle_data, view_range, view_angle, grid_size, grid_height, grid_width):
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    for obstacle_trajectory in obstacle_data:
        if len(obstacle_trajectory) == 1:  # Stationary obstacle
            obstacle = obstacle_trajectory[0]
        else:  # Moving obstacle
            obstacle = obstacle_trajectory[min(i, len(obstacle_trajectory) - 1)]
        
        local_x, local_y = transform_to_local_frame(obstacle['x'], obstacle['y'], agent_x, agent_y, agent_theta)
        
        grid_x = int((local_x + view_range) / grid_size)
        grid_y = int((local_y + view_range) / grid_size)
        
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            if is_within_view(local_x, local_y, view_range, view_angle):
                occupancy_grid[grid_y, grid_x] = 1  # Visible obstacle
            else:
                if occupancy_grid[grid_y, grid_x] == 0:  # Avoid overwriting if already marked as visible
                    occupancy_grid[grid_y, grid_x] = 2  # Invisible obstacle
    return occupancy_grid

# Generate and compare trajectories
def generate_trajectories(model, diffusion, initial_state, goal, obstacle_data, num_trajectories=3, noise_scale=0.1):
    model.eval()
    predicted_trajectories = []

    agent_x, agent_y, agent_theta = initial_state
    goal_x, goal_y = goal

    with torch.no_grad():
        for _ in range(num_trajectories):
            x, y, theta = agent_x, agent_y, agent_theta
            trajectory = [(x, y, theta)]
            for i in range(len(test_trajectory_data) * 2):  # limit the length of the generated trajectory
                occupancy_grid = generate_dynamic_occupancy_map(x, y, theta, obstacle_data, view_range, view_angle, grid_size, grid_height, grid_width)
                occupancy_grid = torch.tensor(occupancy_grid, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Downsample and flatten occupancy_grid before feeding into the model
                downsampled_grid = downsample(occupancy_grid.cpu().numpy()[0])
                downsampled_grid_flat = downsampled_grid.flatten()
                downsampled_grid_flat_tensor = torch.tensor(downsampled_grid_flat, dtype=torch.float32).unsqueeze(0).to(device)
                
                t_step = torch.randint(0, diffusion.num_steps, (1,)).to(device)
                noisy_inputs, noise = diffusion.add_noise(downsampled_grid_flat_tensor, t_step)
                noise_pred = model(noisy_inputs)
                action_pred = noise_pred.view(-1).cpu().numpy()
                
                delta_x, delta_y, delta_theta = action_pred
                delta_x *= max_delta_x
                delta_y *= max_delta_y
                delta_theta *= max_delta_theta

                # Add noise to the predicted action
                delta_x += np.random.normal(0, noise_scale * max_delta_x)
                delta_y += np.random.normal(0, noise_scale * max_delta_y)
                delta_theta += np.random.normal(0, noise_scale * max_delta_theta)
                
                x += delta_x
                y += delta_y
                theta += delta_theta
                
                trajectory.append((x, y, theta))
                
                # Stop if close to the goal
                if np.linalg.norm([x - goal_x, y - goal_y]) < 30:
                    break
            
            predicted_trajectories.append(trajectory)

    return predicted_trajectories

# Define initial state and goal position
initial_state = (test_trajectory_data[0]['x'], test_trajectory_data[0]['y'], test_trajectory_data[0]['theta'])
goal = (test_trajectory_data[-1]['x'], test_trajectory_data[-1]['y'])

# Generate actual trajectory from test data
actual_trajectory = [(test_trajectory_data[0]['x'], test_trajectory_data[0]['y'], test_trajectory_data[0]['theta'])]
x, y, theta = test_trajectory_data[0]['x'], test_trajectory_data[0]['y'], test_trajectory_data[0]['theta']
for action in generate_occupancy_maps_and_actions_with_obstacles(test_trajectory_data, test_obstacle_data)[1]:
    delta_x, delta_y, delta_theta = action
    x += delta_x
    y += delta_y
    theta += delta_theta
    actual_trajectory.append((x, y, theta))

# Generate predicted trajectories
predicted_trajectories = generate_trajectories(model, diffusion, initial_state, goal, test_obstacle_data, noise_scale=0.5)

# Visualization parameters
dot_size = 100

# Determine the fixed range for the global view
all_x_positions = [pos['x'] for pos in test_trajectory_data]
all_y_positions = [pos['y'] for pos in test_trajectory_data]

global_x_min, global_x_max = min(all_x_positions), max(all_x_positions)
global_y_min, global_y_max = min(all_y_positions), max(all_y_positions)

# Set the fixed range for the local view
local_x_min, local_x_max = -view_range, view_range
local_y_min, local_y_max = -view_range, view_range

# Create the animation with separate views for each trajectory
fig, axs = plt.subplots(len(predicted_trajectories) + 1, 2, figsize=(12, 6 * (len(predicted_trajectories) + 1)))

def update(idx):
    for ax in axs.flatten():
        ax.clear()

    # Plot actual trajectory in global view
    actual_xs, actual_ys = zip(*[(x, y) for x, y, theta in actual_trajectory])
    axs[0, 1].plot(actual_xs, actual_ys, label='Actual Trajectory', marker='o', color='blue')

    # Plot the actual trajectory agent's view in local view
    agent_x, agent_y, agent_theta = test_trajectory_data[idx]['x'], test_trajectory_data[idx]['y'], test_trajectory_data[idx]['theta']
    obstacles_in_view = []
    non_viewable_obstacles = []
    global_obstacles_in_view = []
    global_obstacles_non_viewable = []

    for obstacle_trajectory in test_obstacle_data:
        if len(obstacle_trajectory) == 1:
            obstacle = obstacle_trajectory[0]
        else:
            obstacle = obstacle_trajectory[min(idx, len(obstacle_trajectory) - 1)]
        
        local_x, local_y = transform_to_local_frame(obstacle['x'], obstacle['y'], agent_x, agent_y, agent_theta)
        global_x, global_y = obstacle['x'], obstacle['y']
        
        if is_within_view(local_x, local_y, view_range, view_angle):
            obstacles_in_view.append((local_x, local_y))
            global_obstacles_in_view.append((global_x, global_y))
        else:
            non_viewable_obstacles.append((local_x, local_y))
            global_obstacles_non_viewable.append((global_x, global_y))

    if obstacles_in_view:
        xs, ys = zip(*obstacles_in_view)
        axs[0, 0].scatter(xs, ys, s=dot_size, c='red', label='Viewable')
    if non_viewable_obstacles:
        xs, ys = zip(*non_viewable_obstacles)
        axs[0, 0].scatter(xs, ys, s=dot_size, c='gray', label='Non-viewable')
    axs[0, 0].legend()

    # Plot agent's view in local view
    axs[0, 0].set_xlim(local_x_min, local_x_max)
    axs[0, 0].set_ylim(local_y_min, local_y_max)
    axs[0, 0].set_title('Actual Trajectory Agent View')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    
    # Draw the view cone in local view
    wedge = Wedge((0, 0), view_range, -view_angle/2, view_angle/2, alpha=0.2, color='blue')
    axs[0, 0].add_patch(wedge)

    # Plot actual trajectory in global view
    axs[0, 1].scatter(agent_x, agent_y, s=dot_size, c='blue', marker='o', label='Agent')
    axs[0, 1].legend()
    
    # Draw the view cone in global view
    wedge_global_start = transform_to_global_frame(view_range * np.cos(np.deg2rad(-view_angle/2)), view_range * np.sin(np.deg2rad(-view_angle/2)), agent_x, agent_y, agent_theta)
    wedge_global_end = transform_to_global_frame(view_range * np.cos(np.deg2rad(view_angle/2)), view_range * np.sin(np.deg2rad(view_angle/2)), agent_x, agent_y, agent_theta)
    axs[0, 1].plot([agent_x, wedge_global_start[0]], [agent_y, wedge_global_start[1]], 'b--')
    axs[0, 1].plot([agent_x, wedge_global_end[0]], [agent_y, wedge_global_end[1]], 'b--')
    axs[0, 1].add_patch(Wedge((agent_x, agent_y), view_range, agent_theta * 180/np.pi - view_angle/2, agent_theta * 180/np.pi + view_angle/2, alpha=0.2, color='blue'))

    if global_obstacles_in_view:
        xs, ys = zip(*global_obstacles_in_view)
        axs[0, 1].scatter(xs, ys, s=dot_size, c='red', label='Viewable')
    if global_obstacles_non_viewable:
        xs, ys = zip(*global_obstacles_non_viewable)
        axs[0, 1].scatter(xs, ys, s=dot_size, c='gray', label='Non-viewable')
    axs[0, 1].set_xlim(global_x_min, global_x_max)
    axs[0, 1].set_ylim(global_y_min, global_y_max)
    axs[0, 1].scatter(agent_x, agent_y, s=dot_size, c='blue', marker='o', label='Agent')  # Agent's position
    axs[0, 1].legend()

    # Plot the moving actual trajectory up to the current index
    if idx < len(actual_xs):
        axs[0, 1].plot(actual_xs[:idx+1], actual_ys[:idx+1], label='Actual Trajectory', marker='o', color='blue')

    # Plot predicted trajectories and their agent views
    for i, predicted_trajectory in enumerate(predicted_trajectories):
        predicted_xs, predicted_ys = zip(*[(x, y) for x, y, theta in predicted_trajectory])
        
        # Plot the predicted trajectory in global view
        axs[i+1, 1].plot(predicted_xs, predicted_ys, label=f'Predicted Trajectory {i+1}', marker='x')

        # Plot agent's view in local view
        agent_x, agent_y, agent_theta = predicted_trajectory[idx][:3]
        obstacles_in_view = []
        non_viewable_obstacles = []
        global_obstacles_in_view = []
        global_obstacles_non_viewable = []

        for obstacle_trajectory in test_obstacle_data:
            if len(obstacle_trajectory) == 1:
                obstacle = obstacle_trajectory[0]
            else:
                obstacle = obstacle_trajectory[min(idx, len(obstacle_trajectory) - 1)]
            
            local_x, local_y = transform_to_local_frame(obstacle['x'], obstacle['y'], agent_x, agent_y, agent_theta)
            global_x, global_y = obstacle['x'], obstacle['y']
            
            if is_within_view(local_x, local_y, view_range, view_angle):
                obstacles_in_view.append((local_x, local_y))
                global_obstacles_in_view.append((global_x, global_y))
            else:
                non_viewable_obstacles.append((local_x, local_y))
                global_obstacles_non_viewable.append((global_x, global_y))

        if obstacles_in_view:
            xs, ys = zip(*obstacles_in_view)
            axs[i+1, 0].scatter(xs, ys, s=dot_size, c='red', label='Viewable')
        if non_viewable_obstacles:
            xs, ys = zip(*non_viewable_obstacles)
            axs[i+1, 0].scatter(xs, ys, s=dot_size, c='gray', label='Non-viewable')
        axs[i+1, 0].legend()

        axs[i+1, 0].set_xlim(local_x_min, local_x_max)
        axs[i+1, 0].set_ylim(local_y_min, local_y_max)
        axs[i+1, 0].set_title(f'Predicted Trajectory {i+1} Agent View')
        axs[i+1, 0].set_xlabel('X')
        axs[i+1, 0].set_ylabel('Y')
        
        wedge = Wedge((0, 0), view_range, -view_angle/2, view_angle/2, alpha=0.2, color='blue')
        axs[i+1, 0].add_patch(wedge)

        axs[i+1, 1].set_xlim(global_x_min, global_x_max)
        axs[i+1, 1].set_ylim(global_y_min, global_y_max)
        axs[i+1, 1].scatter(agent_x, agent_y, s=dot_size, c='blue', marker='o', label='Agent')
        axs[i+1, 1].legend()
        
        wedge_global_start = transform_to_global_frame(view_range * np.cos(np.deg2rad(-view_angle/2)), view_range * np.sin(np.deg2rad(-view_angle/2)), agent_x, agent_y, agent_theta)
        wedge_global_end = transform_to_global_frame(view_range * np.cos(np.deg2rad(view_angle/2)), view_range * np.sin(np.deg2rad(view_angle/2)), agent_x, agent_y, agent_theta)
        axs[i+1, 1].plot([agent_x, wedge_global_start[0]], [agent_y, wedge_global_start[1]], 'b--')
        axs[i+1, 1].plot([agent_x, wedge_global_end[0]], [agent_y, wedge_global_end[1]], 'b--')
        axs[i+1, 1].add_patch(Wedge((agent_x, agent_y), view_range, agent_theta * 180/np.pi - view_angle/2, agent_theta * 180/np.pi + view_angle/2, alpha=0.2, color='blue'))

        if global_obstacles_in_view:
            xs, ys = zip(*global_obstacles_in_view)
            axs[i+1, 1].scatter(xs, ys, s=dot_size, c='red', label='Viewable')
        if global_obstacles_non_viewable:
            xs, ys = zip(*global_obstacles_non_viewable)
            axs[i+1, 1].scatter(xs, ys, s=dot_size, c='gray', label='Non-viewable')

ani = animation.FuncAnimation(fig, update, frames=len(test_trajectory_data), repeat=False)

# Save the animation as a MP4 file using ffmpeg writer
animation_filename = 'animation.mp4'
try:
    ani.save(animation_filename, writer='ffmpeg', fps=10)
    print(f"Animation saved as {animation_filename} on the local machine.")
except Exception as e:
    print(f"Failed to save animation as MP4 due to: {e}. Trying to save as GIF instead.")
    animation_filename = 'animation.gif'
    ani.save(animation_filename, writer='pillow', fps=10)
    print(f"Animation saved as {animation_filename} on the local machine.")

# Display the animation
HTML(ani.to_jshtml())
