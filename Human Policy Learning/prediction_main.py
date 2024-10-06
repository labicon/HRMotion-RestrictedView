import torch
import numpy as np
from pathlib import Path
from model import ClassConditionedUnet
from diffusers import DDPMScheduler
from prediction import load_trained_model, generate_trajectories, save_trajectories_to_txt, plot_trajectories
from data_loader import load_data, normalize_data

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    model_path = 'best_diffusion_model_with_obstacles.pth'
    model = load_trained_model(model_path, device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")

    initial_state = (20, 20, 0.3)
    goal = (1050, 620)

    base_dir = Path('./data')
    _, all_obstacle_data, _ = load_data(base_dir)
    obstacle_data = all_obstacle_data[0]

    _, _, max_x, max_y, max_obstacle_x, max_obstacle_y, max_delta_x, max_delta_y, max_delta_theta = normalize_data(np.array([]), np.array([]))

    predicted_trajectories = generate_trajectories(
        model, noise_scheduler, initial_state, goal, obstacle_data, initial_t=0,
        max_delta_x=max_delta_x, max_delta_y=max_delta_y, max_delta_theta=max_delta_theta,
        max_x=max_x, max_y=max_y, max_obstacle_x=max_obstacle_x, max_obstacle_y=max_obstacle_y,
        num_trajectories=4
        max_steps=600
    )

    # Save trajectories to a text file
    save_trajectories_to_txt(predicted_trajectories, 'predicted_trajectories.txt')
    print("Trajectories have been saved to 'predicted_trajectories.txt'")

    # Plot trajectories
    final_obstacles = [{'x': obs['x'], 'y': obs['y']} for obs in obstacle_data[-1]]
    plot_trajectories(predicted_trajectories, goal, final_obstacles, save_path='predicted_trajectories_plot.png')

if __name__ == "__main__":
    main()