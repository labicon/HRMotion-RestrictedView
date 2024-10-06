import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from data_loader import load_data, normalize_data, generate_state_with_closest_obstacles
from model import ClassConditionedUnet
from training import train_model
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    base_dir = Path('./data')
    all_trajectory_data, all_obstacle_data, _ = load_data(base_dir)
    goal = (1050, 620)

    # Prepare train and validation data
    train_inputs, train_actions = [], []
    for i in range(len(all_trajectory_data) - 1):
        inputs, actions = generate_state_with_closest_obstacles(
            all_trajectory_data[i], all_obstacle_data[i], goal, view_range=480, view_angle=45)
        train_inputs.append(inputs)
        train_actions.append(actions)
    
    train_inputs = np.vstack(train_inputs)
    train_actions = np.vstack(train_actions)

    train_inputs, train_actions, max_x, max_y, max_obstacle_x, max_obstacle_y, max_delta_x, max_delta_y, max_delta_theta = normalize_data(train_inputs, train_actions)

    train_inputs, val_inputs, train_actions, val_actions = train_test_split(
        train_inputs, train_actions, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(train_inputs, dtype=torch.float32), torch.tensor(train_actions, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_inputs, dtype=torch.float32), torch.tensor(val_actions, dtype=torch.float32))

    model = ClassConditionedUnet(location_emb_size=256).to(device)
    model.device = device  # Set the device attribute for the model
    model_path = 'best_diffusion_model_with_obstacles.pth'

    noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
    ema_power = 0.75
    ema = EMAModel(parameters=model.parameters(), power=ema_power)

    train_model(model, train_dataset, val_dataset, noise_scheduler, ema, model_path)

if __name__ == "__main__":
    main()