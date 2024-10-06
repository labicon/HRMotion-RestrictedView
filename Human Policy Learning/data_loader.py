import json
import numpy as np
import glob

def load_data(base_dir):
    """Load trajectory and obstacle data from the given base directory."""
    data_directories = []
    for p in base_dir.glob('*/*/*/*/'):
        str_path = str(p)
        if '2' in str_path.split('\\')[2]:
            continue
        str_path += '\\'
        data_directories.append(str_path)
    
    all_trajectory_data = []
    all_obstacle_data = []
    trajectory_sources = []
    
    for data_directory in data_directories:
        trajectory_data, obstacle_data = load_data_from_directory(data_directory)
        all_trajectory_data.append(trajectory_data)
        all_obstacle_data.append(obstacle_data)
        trajectory_sources.append(data_directory)
    
    return all_trajectory_data, all_obstacle_data, trajectory_sources

def load_data_from_directory(data_directory):
    """Load data from a specific directory."""
    trajectory_file = data_directory + 'trajectory_data.json'
    with open(trajectory_file) as f:
        trajectory_data = json.load(f)
    
    for step in trajectory_data:
        step['theta'] = np.radians(step['theta'])  # Convert to radians
    
    obstacle_files = sorted(glob.glob(data_directory + 'obstacle*_trajectory_data.json'))
    obstacle_data = []
    for file in obstacle_files:
        with open(file) as f:
            obstacle_data.append(json.load(f))
    
    return trajectory_data, obstacle_data

def normalize_data(train_inputs, train_actions):
    """Normalize input and action data."""
    max_x = np.max(np.abs(train_inputs[:, 0]))
    max_y = np.max(np.abs(train_inputs[:, 1]))

    train_inputs[:, 0] /= max_x
    train_inputs[:, 1] /= max_y
    train_inputs[:, 2] /= np.pi

    obstacle_x_columns = train_inputs[:, 3::2]
    obstacle_y_columns = train_inputs[:, 4::2]

    max_obstacle_x = np.max(np.abs(obstacle_x_columns))
    max_obstacle_y = np.max(np.abs(obstacle_y_columns))

    train_inputs[:, 3::2] /= max_obstacle_x
    train_inputs[:, 4::2] /= max_obstacle_y

    max_delta_x = np.max(np.abs(train_actions[:, 0]))
    max_delta_y = np.max(np.abs(train_actions[:, 1]))
    max_delta_theta = np.pi

    train_actions[:, 0] /= max_delta_x
    train_actions[:, 1] /= max_delta_y
    train_actions[:, 2] /= max_delta_theta

    return train_inputs, train_actions, max_x, max_y, max_obstacle_x, max_obstacle_y, max_delta_x, max_delta_y, max_delta_theta

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

def generate_state_with_closest_obstacles(agent_trajectory, obstacle_data, goal, view_range, view_angle):
    """Create input array from trajectory and obstacle data."""
    inputs = []
    actions = []

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