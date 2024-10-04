import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from diffusers.optimization import get_scheduler
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from torch.utils.tensorboard import SummaryWriter
import glob
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# Define the flag to determine whether to train or load the model
train_model = False   # Set to True if you want to train the model, False to load the existing model

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
    './data/monroe_data/1/96/2024-03-05T164956/',
    './data/monroe_data/1/192/2024-02-27T170725/',
    './data/monroe_data/1/192/2024-03-05T165017/',
    './data/monroe_data/1/192/2024-03-05T165045/',
    './data/monroe_data/1/192/2024-03-05T165102/',
    './data/monroe_data/1/192/2024-03-05T165126/',
    './data/monroe_data/1/192/2024-03-05T165143/',
    './data/monroe_data/1/192/2024-03-05T165204/',
    './data/shivani_data/1/96/2024-03-27T234645/',
    './data/shivani_data/1/96/2024-03-27T234715/',
    './data/shivani_data/1/96/2024-03-27T234742/',
    './data/shivani_data/1/96/2024-03-27T234804/',
    './data/shivani_data/1/96/2024-03-27T234824/',
    './data/shivani_data/1/192/2024-03-27T234858/',
    './data/shivani_data/1/192/2024-03-27T234918/',
    './data/shivani_data/1/192/2024-03-27T234939/',
    './data/shivani_data/1/192/2024-03-27T234957/',
    './data/shivani_data/1/192/2024-03-27T235023/',
    './data/shivani_data/1/480/2024-03-27T235037/',
    './data/shivani_data/1/480/2024-03-27T235054/',
    './data/shivani_data/1/480/2024-03-27T235111/',
    './data/shivani_data/1/480/2024-03-27T235128/',
    './data/shivani_data/1/480/2024-03-27T235145/',
]


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

# Use the first dataset for testing
test_trajectory_data = all_trajectory_data[0]
test_obstacle_data = all_obstacle_data[0]
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

# Generate training data with obstacle and goal information
train_inputs = []
train_actions = []

goal = (all_trajectory_data[0][-1]['x'], all_trajectory_data[0][-1]['y'])

for i in range(len(all_trajectory_data) - 1):
    inputs, actions = generate_state_with_closest_obstacles(
        all_trajectory_data[i], all_obstacle_data[i], goal, history_length, view_range, view_angle)

    train_inputs.append(inputs)
    train_actions.append(actions)

# Convert lists to arrays
train_inputs = np.vstack(train_inputs)
train_actions = np.vstack(train_actions)

print("Before Normalization:", train_inputs[100])

max_x = np.max(np.abs(train_inputs[:, 0]))
max_y = np.max(np.abs(train_inputs[:, 1]))

# print(max_x)
# print(max_y)
train_inputs[:, 0] /= max_x
train_inputs[:, 1] /= max_y
train_inputs[:, 2] /= np.pi


# Normalize obstacle coordinates
# Assuming obstacle coordinates start from column 3 onwards and are interleaved as [obs_x1, obs_y1, obs_x2, obs_y2, ...]
obstacle_x_columns = train_inputs[:, 3::2]  # Every second column starting from 3 corresponds to obstacle x coordinates
obstacle_y_columns = train_inputs[:, 4::2]  # Every second column starting from 4 corresponds to obstacle y coordinates

max_obstacle_x = np.max(np.abs(obstacle_x_columns))
max_obstacle_y = np.max(np.abs(obstacle_y_columns))

# Normalize the obstacle coordinates
train_inputs[:, 3::2] /= max_obstacle_x  # Normalize obstacle x coordinates
train_inputs[:, 4::2] /= max_obstacle_y  # Normalize obstacle y coordinates

max_delta_x = np.max(np.abs(train_actions[:, 0]))
max_delta_y = np.max(np.abs(train_actions[:, 1]))
max_delta_theta = np.pi

print('max_delta_x', max_delta_x)
# print(max_delta_y)
# print(np.mean(train_actions[:, 0]))
# print(np.mean(train_actions[:, 1]))
# print(np.min(np.abs(train_actions[:, 0])))
# print(np.min(np.abs(train_actions[:, 1])))
print('max_obstacle_x', max_obstacle_x)
print('max_obstacle_y', max_obstacle_y)
train_actions[:, 0] /= max_delta_x
train_actions[:, 1] /= max_delta_y
train_actions[:, 2] /= max_delta_theta

print("After Normalization:", train_inputs[100])

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

# Normalize the test obstacles
normalized_test_obstacle_data = normalize_obstacles(test_obstacle_data, max_obstacle_x, max_obstacle_y)

# Split data into training and validation sets
train_inputs, val_inputs, train_actions, val_actions = train_test_split(
    train_inputs, train_actions, test_size=0.2, random_state=42)

# Define the ClassConditionedUnet model using UNet2DConditionModel
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
        self.location_emb = nn.Linear(9, location_emb_size)  # Adjust input size based on your context (e.g., location and obstacles)

        # MLP to project noisy_action to a higher spatial dimension
        self.project_to_high_dim = MLP(3, hidden_dim, 3 * output_size * output_size)  # Input 3, hidden_dim intermediate, output 3 * H * W
        self.reduce_to_output = nn.Linear(3 * output_size * output_size, 3)

        # Define the conditional UNet model with cross-attention
        self.model = UNet2DConditionModel(
            sample_size=output_size,  # The input image size (e.g., 16x16 after projection)
            in_channels=3,  # The number of input channels (noisy action)
            out_channels=3,  # The number of output channels (predicted noise)
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=(
                "CrossAttnDownBlock2D",  # Use cross-attention in the downsampling path
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",  # Use cross-attention in the upsampling path
                "CrossAttnUpBlock2D",  
                "CrossAttnUpBlock2D",  # You can keep the final block as standard or use cross-attention here as well
            ),
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

# Filepath for saving/loading the model
model_path = 'best_diffusion_model_with_obstacles.pth'

noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")

ema = EMAModel(
        parameters=model.parameters(),
        power=0.75
    )

if train_model:
    # Training loop
    num_epochs = 100
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(torch.tensor(train_inputs, dtype=torch.float32), torch.tensor(train_actions, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_inputs, dtype=torch.float32), torch.tensor(val_actions, dtype=torch.float32))

    batch_size = 256
    accumulation_steps = 2
    best_val_loss = float('inf')
    early_stopping_patience = 20
    patience_counter = 0

    scaler = GradScaler()
    writer = SummaryWriter()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=4e-4, weight_decay=1e-6)
    
    # Set up the Cosine Learning Rate Scheduler with Linear Warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * num_epochs
    )

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for i, (inputs, actions) in enumerate(train_loader):
            inputs = inputs.to(device)
            actions = actions.to(device)

            # Sample random timesteps for the diffusion process
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (inputs.size(0),)).to(device)
            noise = torch.randn_like(actions)  # Generate random noise
            noisy_actions = noise_scheduler.add_noise(actions, noise, t)  # Add noise to actions

            # Use inputs as location conditioning
            location = inputs
            noisy_inputs = noisy_actions

            with autocast():  # Mixed precision training
                noise_pred = model(noisy_inputs, t, location)
                loss = criterion(noise_pred, noise)  # Calculate loss

            # Backward pass
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping

            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update EMA model parameters after optimizer step
                ema.step(model.parameters())

                # Update learning rate scheduler
                lr_scheduler.step()

            train_loss += loss.item() * inputs.size(0)

        # Handle the final gradient step if not a multiple of accumulation_steps
        if (i + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update EMA model parameters after optimizer step
            ema.step(model.parameters())

            # Update learning rate scheduler
            lr_scheduler.step()

        # Average train loss over the dataset
        train_loss /= len(train_loader.dataset)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, actions in val_loader:
                inputs = inputs.to(device)
                actions = actions.to(device)

                t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (inputs.size(0),)).to(device)
                noise = torch.randn_like(actions)
                noisy_actions = noise_scheduler.add_noise(actions, noise, t)
                location = inputs
                noisy_inputs = noisy_actions

                with autocast():  # Mixed precision during validation
                    noise_pred = model(noisy_inputs, t, location)
                    loss = criterion(noise_pred, noise)

                val_loss += loss.item() * inputs.size(0)

        # Average validation loss over the dataset
        val_loss /= len(val_loader.dataset)
        writer.add_scalar('Loss/val', val_loss, epoch)

        # Early stopping and model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the best model checkpoint, including EMA state
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping")
                break

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Close the TensorBoard writer
    writer.close()

    # Store and use the EMA model for inference
    ema.store(model.parameters())  # Store current parameters
    ema.copy_to(model.parameters())  # Use EMA parameters for evaluation

    print(f"Training complete. Model saved to {model_path}")

else:
    # Load the pre-trained model
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])  # Load EMA state
        print(f"Model loaded from {model_path}")
    else:
        print(f"No pre-trained model found at {model_path}. Please set `train_model = True` to train a new model.")

# Function to generate trajectories using the trained model and the scheduler
def generate_trajectories(model, noise_scheduler, initial_state, goal, obstacle_data, initial_t, max_delta_x, max_delta_y, max_delta_theta, max_x, max_y, max_obstacle_x, max_obstacle_y, num_trajectories=3, max_steps=500):
    model.eval()
    predicted_trajectories = []

    agent_x, agent_y, agent_theta = initial_state
    agent_x /= max_x
    agent_y /= max_y 
    agent_theta /= np.pi
    goal_x, goal_y = goal

    # Move scheduler's internal tensors to GPU
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    with torch.no_grad():
        for _ in range(num_trajectories):
            x, y, theta = agent_x, agent_y, agent_theta
            trajectory = [(x * max_x, y * max_y, theta * np.pi)]

            for i in tqdm(range(max_steps)):
                current_t = initial_t + i
                closest_obstacles = generate_closest_obstacles(x * max_x, y * max_y, theta * np.pi, obstacle_data, view_range, view_angle, current_t)

                closest_obstacles = np.array(closest_obstacles, dtype=np.float32)

                # Normalize obstacles
                closest_obstacles[0::2] /= max_obstacle_x  # Normalize x-coordinates of obstacles
                closest_obstacles[1::2] /= max_obstacle_y  # Normalize y-coordinates of obstacles

                # Concatenate the agent's state with normalized obstacles
                state_with_obstacles = np.concatenate(([x, y, theta], closest_obstacles))
                state_tensor = torch.tensor(state_with_obstacles, dtype=torch.float32).unsqueeze(0).to(device)

                # Generate a noisy action (with batch dimension)
                random_action = torch.randn(1, 3).to(device)

                # Expand random_action to match the UNet output shape [batch_size, 3, 32, 32]
                #random_action_expanded = random_action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 32)

                #random_action_expanded = random_action.view(random_action.size(0), random_action.size(1), 1, 1)
                #print(random_action_expanded.shape)

                random_action_expanded = random_action

                for _, step in enumerate(noise_scheduler.timesteps):
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
                delta_x, delta_y, delta_theta = action_pred

                # print('delta_x', delta_x)
                # print('delta_y', delta_y)
                # print('delta_theta', delta_theta)

                de_x = x * max_x
                de_y = y * max_y
                de_theta = theta * np.pi

                a_x = delta_x * max_delta_x
                a_y = delta_y * max_delta_y
                a_theta = delta_theta * max_delta_theta

                de_x += a_x * 7
                de_y += a_y * 7
                de_theta += a_theta * 7

                if de_theta >  np.pi:
                    de_theta -= 2 * np.pi
                elif de_theta < -np.pi:
                    de_theta += 2 * np.pi
                
                # print(de_x)
                # print(de_y)
                # print(de_theta)
                trajectory.append((de_x, de_y, de_theta))

                x = de_x / max_x
                y = de_y / max_y
                theta = de_theta/ np.pi

                if np.linalg.norm([de_x - goal_x, de_y - goal_y]) < 60:
                    print('Reaching goal')
                    print('Step required', i)
                    break
                if de_x >= 1300 or de_y > 700 or de_y < -100 or de_x < 0:
                    print('Agent out of range')
                    break
            print('trajectory', i)
            predicted_trajectories.append(trajectory)

    return predicted_trajectories

# Generate predicted trajectories
initial_t = 0
initial_state = (200, 200, 0.3)
goal = (test_trajectory_data[-1]['x'], test_trajectory_data[-1]['y'])

# Create a cluster of obstacles around (500, 400)
cluster_center_x = 500
cluster_center_y = 400
cluster_radius = 100  # Define a radius for obstacle distribution
num_cluster_obstacles = 60  # Number of obstacles in the cluster

# Randomly distribute obstacles within the cluster radius
np.random.seed(42)  # For reproducibility
cluster_obstacles = [
    {
        'x': cluster_center_x + np.random.uniform(-cluster_radius, cluster_radius),
        'y': cluster_center_y + np.random.uniform(-cluster_radius, cluster_radius)
    }
    for _ in range(num_cluster_obstacles)
]

# Add the cluster of obstacles to the test_obstacle_data
test_obstacle_data.extend([ [obs] for obs in cluster_obstacles ])  # Treat each as a stationary obstacle



predicted_trajectories = generate_trajectories(model, noise_scheduler, initial_state, goal, test_obstacle_data, initial_t, max_delta_x, max_delta_y, max_delta_theta, max_x, max_y, max_obstacle_x, max_obstacle_y, max_steps=500)

def save_trajectories_to_txt(trajectories, filename):
    with open(filename, 'w') as file:
        for i, trajectory in enumerate(trajectories):
            file.write(f"Trajectory {i+1}:\n")
            for x, y, theta in trajectory:
                file.write(f"{x}, {y}, {theta}\n")
            file.write("\n")  # Add a blank line between trajectories

# Save the predicted trajectories to a text file
save_trajectories_to_txt(predicted_trajectories, 'predicted_trajectories.txt')

print("Trajectories have been saved to 'predicted_trajectories.txt'")

def plot_trajectories(predicted_trajectories, goal, final_obstacles, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.plot(goal[0], goal[1], 'go', markersize=10, label='Goal')

    for i, trajectory in enumerate(predicted_trajectories):
        x_vals = [state[0] for state in trajectory]
        y_vals = [state[1] for state in trajectory]
        plt.plot(x_vals, y_vals, label=f'Trajectory {i+1}')

    obstacle_x_vals = [obs['x'] for obs in final_obstacles]
    obstacle_y_vals = [obs['y'] for obs in final_obstacles]
    plt.plot(obstacle_x_vals, obstacle_y_vals, 'ro', markersize=5, label='Final Obstacles')

    plt.title('Predicted Trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

# Define the path to save the plot
save_path = 'predicted_trajectories_plot.png'
final_obstacles = [obstacle[-1] for obstacle in test_obstacle_data]

# Plot and save the predicted trajectories
plot_trajectories(predicted_trajectories, goal, final_obstacles, save_path=save_path)
