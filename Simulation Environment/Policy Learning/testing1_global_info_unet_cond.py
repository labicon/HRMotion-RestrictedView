import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from diffusers import DDPMScheduler, UNet2DConditionModel
import glob
import os
import matplotlib.pyplot as plt

# Define the flag to determine whether to train or load the model
train_model = False  # Set to True if you want to train the model, False to load the existing model

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

# Parameters
view_range = 480  # Adjust to 192 or 480 as needed
view_angle = 45   # View angle in degrees
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
        combined_input = [agent_x, agent_y, agent_theta]
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

# Split data into training and validation sets
train_inputs, val_inputs, train_actions, val_actions = train_test_split(
    train_inputs, train_actions, test_size=0.2, random_state=42)

# Define the ClassConditionedUnet model using UNet2DConditionModel
class ClassConditionedUnet(nn.Module):
    def __init__(self, location_emb_size=1280):  # Set to match cross_attention_dim
        super().__init__()
        
        # Linear layer to map (x, y, theta) to a vector of size location_emb_size
        self.location_emb = nn.Linear(3, location_emb_size)  # Project to the correct dimension

        # Define the conditional UNet model
        self.model = UNet2DConditionModel(
            sample_size=32,  # The input image size (32x32)
            in_channels=3,  # The number of input channels (noisy action)
            out_channels=3,  # The number of output channels (predicted noise)
            layers_per_block=2,
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D", 
                "AttnDownBlock2D", 
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D", 
                "AttnUpBlock2D",  
                "UpBlock2D",  
            ),
            cross_attention_dim=location_emb_size,  # The dimension of the cross-attention
        )

    def forward(self, noisy_action, t, location):
        # Expand noisy_action to have spatial dimensions of 32x32 (e.g., [batch_size, 3, 32, 32])
        if noisy_action.size(-1) != 32 or noisy_action.size(-2) != 32:
        # Expand noisy_action to have spatial dimensions of 32x32 (e.g., [batch_size, 3, 32, 32])
            noisy_action = noisy_action.view(noisy_action.size(0), noisy_action.size(1), 1, 1)
            noisy_action = F.interpolate(noisy_action, size=(32, 32))  # Resize to 32x32

        # Embed the current location (x, y, theta)
        location_emb = self.location_emb(location)  # Shape: (batch_size, location_emb_size)

        # Reshape location_emb to have a spatial dimension (e.g., [batch_size, 1024, location_emb_size])
        location_emb = location_emb.unsqueeze(1).repeat(1, 1024, 1)  # Repeat to create spatial dimension

        # Pass through the UNet model
        return self.model(noisy_action, timestep=t, encoder_hidden_states=location_emb).sample  # Predict the noise for all 3 action components


# Initialize the model
model = ClassConditionedUnet(location_emb_size=3).to(device)

# Filepath for saving/loading the model
model_path = 'best_diffusion_model_with_obstacles.pth'

noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")

if train_model:
    # Training loop
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

    train_dataset = TensorDataset(torch.tensor(train_inputs, dtype=torch.float32), torch.tensor(train_actions, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_inputs, dtype=torch.float32), torch.tensor(val_actions, dtype=torch.float32))

    batch_size = 256
    accumulation_steps = 2
    num_epochs = 20
    best_val_loss = float('inf')
    early_stopping_patience = 20
    patience_counter = 0

    scaler = GradScaler()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, actions) in enumerate(train_loader):
            inputs = inputs.to(device)
            actions = actions.to(device)

            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (inputs.size(0),)).to(device)
            noise = torch.randn_like(actions)  # Shape: [batch_size, 3]
            noisy_actions = noise_scheduler.add_noise(actions, noise, t)

            location = inputs  # Using the inputs as the location conditioning
            noisy_inputs = noisy_actions.view(noisy_actions.size(0), noisy_actions.size(1), 1, 1)  # Reshape to [batch_size, 3, 1, 1]

            with autocast():
                noise_pred = model(noisy_inputs, t, location)  # Shape: [batch_size, 3, 32, 32]

                # Expand the ground truth noise to match the shape of noise_pred
                noise_expanded = noise.view(noise.size(0), noise.size(1), 1, 1).expand(-1, -1, 32, 32)  # Shape: [batch_size, 3, 32, 32]

                # Calculate MSE loss
                loss = criterion(noise_pred, noise_expanded)

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * inputs.size(0)
        
        if (i + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, actions in val_loader:
                inputs = inputs.to(device)
                actions = actions.to(device)

                t = torch.randint(0, noise_scheduler.num_train_timesteps, (inputs.size(0),)).to(device)
                noise = torch.randn_like(actions)
                noisy_actions = noise_scheduler.add_noise(actions, noise, t)

                location = inputs  # Using the inputs as the location conditioning
                noisy_inputs = noisy_actions.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions

                with autocast():
                    noise_pred = model(noisy_inputs, t, location)
                    loss = criterion(noise_pred, noise.view(noise.size(0), noise.size(1), 1, 1).expand(-1, -1, 32, 32))
                
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping")
                break
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    print(f"Training complete. Model saved to {model_path}")

else:
    # Load the pre-trained model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print(f"No pre-trained model found at {model_path}. Please set `train_model = True` to train a new model.")

# Function to generate trajectories using the trained model and the scheduler
def generate_trajectories(model, noise_scheduler, initial_state, goal, obstacle_data, initial_t, history_length=100, num_trajectories=30, max_steps=500):
    model.eval()
    predicted_trajectories = []

    agent_x, agent_y, agent_theta = initial_state
    goal_x, goal_y = goal

    # Move scheduler's internal tensors to GPU
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    with torch.no_grad():
        for _ in range(num_trajectories):
            x, y, theta = agent_x, agent_y, agent_theta
            trajectory = [(x, y, np.degrees(theta))]
            history = []

            for i in range(max_steps):
                if len(history) < history_length * 3:
                    history = [0, 0, 0] * (history_length - len(history) // 3) + history
                else:
                    history.pop(0)
                    history.pop(0)
                    history.pop(0)

                history.extend([x, y, theta])

                current_t = initial_t + i
                closest_obstacles = generate_closest_obstacles(x, y, theta, obstacle_data, view_range, view_angle, current_t)

                state_with_obstacles = [x, y, np.degrees(theta)]
                state_tensor = torch.tensor(state_with_obstacles, dtype=torch.float32).unsqueeze(0).to(device)

                # Generate a noisy action (with batch dimension)
                random_action = torch.randn(1, 3).to(device)

                # Expand random_action to match the UNet output shape [batch_size, 3, 32, 32]
                random_action_expanded = random_action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 32)

                for step in noise_scheduler.timesteps:
                    t_step = torch.tensor([step], dtype=torch.long).to(device)
                    noisy_inputs = random_action_expanded  # Use the expanded version for input
                    #print(noisy_inputs.shape)
                    noise_pred = model(noisy_inputs, t_step, state_tensor)
                    result = noise_scheduler.step(model_output=noise_pred.squeeze(), timestep=t_step, sample=random_action_expanded)
                    random_action_expanded = result.prev_sample  # Extract the previous timestep sample

                # After denoising, take the mean over spatial dimensions to reduce back to shape [1, 3]
                action_pred = random_action_expanded.mean(dim=[2, 3]).squeeze(0).cpu().numpy()
                delta_x, delta_y, delta_theta = action_pred

                x += delta_x * 7
                y += delta_y * 7
                theta += delta_theta

                if theta > np.pi:
                    theta -= 2 * np.pi
                elif theta < -np.pi:
                    theta += 2 * np.pi

                trajectory.append((x, y, np.degrees(theta)))

                if np.linalg.norm([x - goal_x, y - goal_y]) < 30:
                    print('Reaching goal')
                    print('Step require', i)
                    break

            predicted_trajectories.append(trajectory)

    return predicted_trajectories

# Generate predicted trajectories
initial_t = 0
initial_state = (test_trajectory_data[initial_t]['x'], test_trajectory_data[initial_t]['y'], test_trajectory_data[initial_t]['theta'])
goal = (test_trajectory_data[-1]['x'], test_trajectory_data[-1]['y'])

predicted_trajectories = generate_trajectories(model, noise_scheduler, initial_state, goal, test_obstacle_data, initial_t)

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

def plot_trajectories(predicted_trajectories, goal, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.plot(goal[0], goal[1], 'go', markersize=10, label='Goal')

    for i, trajectory in enumerate(predicted_trajectories):
        x_vals = [state[0] for state in trajectory]
        y_vals = [state[1] for state in trajectory]
        plt.plot(x_vals, y_vals, label=f'Trajectory {i+1}')

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

# Plot and save the predicted trajectories
plot_trajectories(predicted_trajectories, goal, save_path=save_path)
