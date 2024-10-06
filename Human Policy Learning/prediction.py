import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from model import ClassConditionedUnet
from diffusers import DDPMScheduler
from data_loader import generate_closest_obstacles

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(model_path, device):
    """Load a trained model from a given path."""
    model = ClassConditionedUnet(location_emb_size=256).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.device = device
    return model


def generate_trajectories(model, noise_scheduler, initial_state, goal, obstacle_data, initial_t, max_delta_x, max_delta_y, max_delta_theta, max_x, max_y, max_obstacle_x, max_obstacle_y, num_trajectories=4, max_steps=600):
    """Generate trajectories based on the trained model."""
    model.eval()
    predicted_trajectories = []

    agent_x, agent_y, agent_theta = initial_state
    agent_x /= max_x
    agent_y /= max_y
    agent_theta /= np.pi
    goal_x, goal_y = goal

    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    variations = [
        (0, 60),
        (10, 10),
        (30, 20),
        (40, 0)
    ]

    with torch.no_grad():
        for variation in variations:
            for _ in range(num_trajectories // len(variations)):
                x, y, theta = agent_x, agent_y, agent_theta
                x = agent_x + variation[0] / max_x
                y = agent_y + variation[1] / max_y
                trajectory = [(x * max_x, y * max_y, theta * np.pi)]

                for i in tqdm(range(max_steps)):
                    current_t = initial_t + i
                    closest_obstacles = generate_closest_obstacles(x * max_x, y * max_y, theta * np.pi, obstacle_data, view_range=480, view_angle=45, time_step=current_t)

                    closest_obstacles = np.array(closest_obstacles, dtype=np.float32)

                    closest_obstacles[0::2] /= max_obstacle_x
                    closest_obstacles[1::2] /= max_obstacle_y

                    state_with_obstacles = np.concatenate(([x, y, theta], closest_obstacles))
                    state_tensor = torch.tensor(state_with_obstacles, dtype=torch.float32).unsqueeze(0).to(device)

                    random_action = torch.randn(1, 3).to(device) * 0.1

                    random_action_expanded = random_action

                    for _, step in enumerate(noise_scheduler.timesteps):
                        t_step = torch.tensor([step], dtype=torch.long).to(device)
                        noisy_inputs = random_action_expanded
                        noise_pred = model(noisy_inputs, t_step, state_tensor)
                        result = noise_scheduler.step(model_output=noise_pred, timestep=t_step, sample=random_action_expanded)
                        random_action_expanded = result.prev_sample

                    action_pred = random_action_expanded.squeeze().cpu()
                    delta_x, delta_y, delta_theta = action_pred

                    de_x = x * max_x
                    de_y = y * max_y
                    de_theta = theta * np.pi

                    a_x = delta_x * max_delta_x
                    a_y = delta_y * max_delta_y
                    a_theta = delta_theta * max_delta_theta

                    de_x += a_x
                    de_y += a_y
                    de_theta += a_theta

                    if de_theta > np.pi:
                        de_theta -= 2 * np.pi
                    elif de_theta < -np.pi:
                        de_theta += 2 * np.pi

                    trajectory.append((de_x, de_y, de_theta))

                    x = de_x / max_x
                    y = de_y / max_y
                    theta = de_theta / np.pi

                    if np.linalg.norm([de_x - goal_x, de_y - goal_y]) < 50:
                        print('Reaching goal')
                        print('Step required', i)
                        break
                    if de_x >= 1200 or de_y > 800 or de_y < -100 or de_x < -50:
                        print('Agent out of range')
                        break
                print('trajectory', i)
                predicted_trajectories.append(trajectory)

    return predicted_trajectories


def save_trajectories_to_txt(trajectories, filename):
    """Save predicted trajectories to a text file."""
    with open(filename, 'w') as file:
        for i, trajectory in enumerate(trajectories):
            file.write(f"Trajectory {i + 1}:
")
            for x, y, theta in trajectory:
                file.write(f"{x}, {y}, {theta}
")
            file.write("\n")


def plot_trajectories(predicted_trajectories, goal, final_obstacles, save_path=None):
    """Plot predicted trajectories along with the goal and obstacles."""
    plt.figure(figsize=(10, 10))
    plt.plot(goal[0], goal[1], 'go', markersize=10, label='Goal')

    for i, trajectory in enumerate(predicted_trajectories):
        x_vals = [state[0] for state in trajectory]
        y_vals = [state[1] for state in trajectory]
        plt.plot(x_vals, y_vals, label=f'Trajectory {i + 1}')

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