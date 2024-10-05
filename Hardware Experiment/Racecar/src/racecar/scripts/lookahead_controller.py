import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

class LookAheadController:
    def __init__(self, lookahead_distance, scale, throttle=60):
        # Initialize ROS node
        rospy.init_node('lookahead_controller', anonymous=True)
        rospy.Subscriber("vrpn_client_node/Racecar/pose", PoseStamped, self.pose_callback)
        self.steer_pub = rospy.Publisher('steer', Float32, queue_size=1)
        self.throttle_pub = rospy.Publisher('throttle', Float32, queue_size=1)

        folder_path = '/home/racecar/Documents/racecar_ws/src/Racecar/src/racecar/scripts/trajs/'
        # traj_name = 'square500.txt'
        # traj_name = 'star500.txt'
        # traj_name = 'circle.txt'
        # traj_name = 'kinsmen500.txt'
        # traj_name = 'lagunaseca1000.txt'
        # traj_name = 'buttonwillow1000.txt'
        traj_name = 'validation1000.txt'
        with open(folder_path+traj_name, 'rb') as f:
            raw_points = np.loadtxt(f, delimiter=' ')
            traj_points = raw_points[:,1:]*scale + np.array([-0.0,-0.5])
        self.traj_points = np.array(traj_points)
        
        self.lookahead_distance = lookahead_distance
        self.L = 0.275
        self.kill = False
        self.throttle = throttle
        self.lap = 0
        self.last_t = 0.0

        self.enable = False
        self.enable_time = 0.0
        self.enable_threshold = 0.1  # maximum 100ms without enable signal

        self.current_pos = None

    def quat_to_heading(self, current_quat):
        current_orient = R.from_quat(current_quat)        
        current_euler  = current_orient.as_euler("YZX")
        heading_vector = R.from_euler("XYZ", [0,current_euler[0],0]).as_matrix()
        heading_vector = heading_vector[[0,2],0] # [x,z] of nose pointing in mocap frame (2D)
        return heading_vector

    def pose_callback(self, data):
        # Update current pose
        self.current_pos = np.array([data.pose.position.x, data.pose.position.z])
        self.current_quat = (data.pose.orientation.x, data.pose.orientation.y,
                             data.pose.orientation.z, data.pose.orientation.w)
        
        self.heading_vec = self.quat_to_heading(self.current_quat)

        if self.current_pos is not None:
            self.control()

    def control(self):
        current_tail_pos = self.current_pos - self.heading_vec*self.L
        # Find the closest point to the vehicle on the trajectory
        closest_idx = np.argmin(np.linalg.norm(self.traj_points - current_tail_pos, axis=1))

        # Search for the lookahead target point along the trajectory
        found = False
        for i in range(closest_idx, len(self.traj_points)):
            target_point = self.traj_points[i]
            if np.linalg.norm(self.traj_points[i] - current_tail_pos) >= self.lookahead_distance:
                found = True
                break
        
        if not found:
            for i in range(0, closest_idx):
                target_point = self.traj_points[i]
                if np.linalg.norm(self.traj_points[i] - current_tail_pos) >= self.lookahead_distance:
                    found = True
                    break

        # Calculate the steering angle based on the target point
        # pure pursuit
        
        tail_to_la_vec = target_point-current_tail_pos
        dist_la = np.linalg.norm(tail_to_la_vec)
        alpha = np.arccos(np.dot(tail_to_la_vec/dist_la, self.heading_vec))
        alpha = alpha*np.sign(np.cross(self.heading_vec,tail_to_la_vec))
        steer_angle = np.arctan2(2*self.L*np.sin(alpha),dist_la)
        # print(target_point)
        # print(np.rad2deg(steer_angle))
        # steer_angle = np.arctan2(target_point[1] - self.current_pos[1], target_point[0] - self.current_pos[0])

        # Publish the steering angle
        steer_command = int(np.rad2deg(steer_angle)/25*128+127)
        steer_command = min(max(steer_command,0),255)
        if not self.kill:
            self.throttle_pub.publish(self.throttle)
            self.steer_pub.publish(steer_command)

        # Lap counter
        dist = np.linalg.norm(np.array(self.current_pos)-self.traj_points[0,0:2])
        curr_t = time.time()
        if (dist < 0.1) and (curr_t - self.last_t > 2):
            self.lap = self.lap+1
            self.last_t = curr_t
            print('Lap Count:',self.lap)

    def shutdown_hook(self):
        print("Shutting down... Setting throttle and steer to zero.")
        self.kill = True
        t_stop = int(self.throttle/60*6)
        self.steer_pub.publish(127)
        for i in range(t_stop):
            self.throttle_pub.publish(-250)
            self.throttle_pub.publish(-250)
            self.throttle_pub.publish(-250)
            time.sleep(0.1)
            self.throttle_pub.publish(-250)
            self.throttle_pub.publish(-250)
        self.throttle_pub.publish(0)
        # self.steer_pub.publish(127)
        time.sleep(0.1)

    def ena_callback(self, data):
        self.enable = data.data > 0.5
        self.enable_time = data.header.stamp

if __name__ == '__main__':
    try:
        # Sample trajectory points
        # traj_points = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

        lookahead_distance = 0.43
        # lookahead_distance = 0.7
        controller = LookAheadController(lookahead_distance, scale=1.5, throttle=60)
        rospy.on_shutdown(controller.shutdown_hook)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
