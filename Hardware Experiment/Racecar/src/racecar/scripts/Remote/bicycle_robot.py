import numpy as np
import math

class BicycleRobot:

    def __init__(self, init_x, init_y, init_theta, init_phi, init_obs, robot_l, robot_b, data):
        self.x = init_x
        self.y = init_y
        self.theta = init_theta
        self.phi = init_phi  # Steering angle
        self.l = robot_l  # Wheelbase (distance between front and rear wheel)
        self.obs = init_obs  # Sensor skirt angle
        self.data = data
        self.b = robot_b
        
        self.X = np.array([self.x, self.y])
        
        self.tip = [self.x + self.l/2 * math.cos(self.theta), self.y + self.l/2 * math.sin(self.theta)]
        self.bottom = [self.x - self.l/2 * math.cos(self.theta), self.y - self.l/2 * math.sin(self.theta)]
        self.bottom_l = [self.bottom[0] - self.b/2 * math.sin(self.theta), self.bottom[1] + self.b/2 * math.cos(self.theta)]
        self.bottom_r = [self.bottom[0] + self.b/2 * math.sin(self.theta), self.bottom[1] - self.b/2 * math.cos(self.theta)]
        self.top_l = [self.tip[0] - self.b/2 * math.sin(self.theta), self.tip[1] + self.b/2 * math.cos(self.theta)]
        self.top_r = [self.tip[0] + self.b/2 * math.sin(self.theta), self.tip[1] - self.b/2 * math.cos(self.theta)]

    # def show(self, color=(255,0,0)):
    #     pygame.draw.polygon(self.data["screen"], color, [self.bottom_l, self.bottom_r, self.top_r, self.top_l], 0)
    #     pygame.draw.polygon(self.data["screen"], color, [self.tip, self.bottom_l, self.bottom_r], 0)
        # self.update_tip_bottom()

    # def update_tip_bottom(self):
    #     # Compute the robot's tip and bottom points
    #     self.tip = [self.x + self.l * math.cos(self.theta), self.y + self.l * math.sin(self.theta)]
    #     self.bottom = [self.x - self.l * math.cos(self.theta), self.y - self.l * math.sin(self.theta)]
    #     self.bottom_l = [self.bottom[0] - self.data["b"] * math.sin(self.theta), self.bottom[1] + self.data["b"] * math.cos(self.theta)]
    #     self.bottom_r = [self.bottom[0] + self.data["b"] * math.sin(self.theta), self.bottom[1] - self.data["b"] * math.cos(self.theta)]
        
    def compute_controls(self, delta_x, delta_y):
        x_target = self.x + delta_x
        y_target = self.y + delta_y

        # Desired heading
        theta_d = math.atan2(y_target - self.y, x_target - self.x)
        distance_to_target = np.sqrt((x_target - self.x)**2 + (y_target - self.y)**2)

        # Desired velocity
        v = min(self.data["vmax"], 10*distance_to_target)

        # Desired steering angle
        phi_d = math.atan2(math.sin(theta_d - self.theta), math.cos(theta_d - self.theta))
        
        #clip the desired steering angle between -pi/3 to pi/3
        phi_d = np.clip(phi_d, -np.pi/3, np.pi/3)

        # Steering angle rate
        phi_error = phi_d - self.phi
        phi_rate = self.data["K_p"] * phi_error

        return v, phi_rate

    def go_to_goal(self):
        e = self.data["goalX"] - self.X  # Error in position
        K = self.data["vmax"] * (1 - np.exp(-self.data["gtg_scaling"] * np.linalg.norm(e)**2)) / np.linalg.norm(e)  # Scaling for velocity
        v = np.linalg.norm(K * e)  # Velocity decreases as bot gets closer to goal
        theta_d = math.atan2(e[1], e[0])  # Desired heading
        delta_theta = math.atan2(math.sin(theta_d - self.theta), math.cos(theta_d - self.theta))  # Error in heading
        omega = self.data["K_p"] * delta_theta  # Only P part of a PID controller for omega
        return [v, omega]

    def avoid_obst(self, obstX, obs_radius):
        e = obstX - self.X  # Error in position
        K = self.data["vmax"] * (1 - np.exp(-self.data["ao_scaling"] * (np.linalg.norm(e) - obs_radius)**2)) / (np.linalg.norm(e) - obs_radius)  # Scaling for velocity
        v = np.linalg.norm(K * e)  # Velocity decreases as bot gets closer to obstacle
        theta_d = math.atan2(-e[0], e[1])  # Desired heading to avoid obstacle
        delta_theta = math.atan2(math.sin(theta_d - self.theta), math.cos(theta_d - self.theta))  # Error in heading
        omega = self.data["K_p"] * delta_theta  # Only P part of a PID controller for omega
        return [v, omega]

    def update_state(self, v, omega, dt):
        # Update state based on control inputs
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += (v / self.l) * math.tan(self.phi) * dt
        self.phi += omega * dt
        self.X = np.array([self.x, self.y])
        self.update_tip_bottom()
