#!/home/racecar/mambaforge/envs/racecar/bin/python3
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
import time 
from scipy.spatial.transform import Rotation as R

class PDController:
    def __init__(self, Kp, Kd, scale=1.0, speed=60, direction="CW"):
        # Initialize
        rospy.init_node('pd_controller')
        rospy.Subscriber("vrpn_client_node/Racecar/pose", PoseStamped, self.pose_callback)
        # rospy.Subscriber("Enable_controller", Float32, self.ena_callback)
        self.steer_pub    = rospy.Publisher('steer', Float32, queue_size=1)
        self.throttle_pub = rospy.Publisher('throttle', Float32, queue_size=1)

        self.Kp = Kp
        self.Kd = Kd
        self.scale = scale
        self.speed  = speed
        if direction == "CW":        
            self.direction = 1
        else:
            self.direction = -1
        self.lap = 0
        self.last_t = 0.0
        self.kill = False

        self.enable = False
        self.enable_time = 0.0
        self.enable_threshold = 0.1  # maximum 100ms without enable signal

        self.error_prev = 0.0
        self.error_dot  = 0.0
        self.error_prev_t = time.time()
        self.start_time = time.time()

        # Define line segments for the trajectory
        folder_path = '/home/racecar/Documents/racecar_ws/src/Racecar/src/racecar/scripts/'
        # traj_name = 'star50.txt'
        # traj_name = 'circle.txt'
        traj_name = 'lagunaseca1000.txt'
        traj_name = 'kinsmen500.txt'
        with open(folder_path+traj_name, 'rb') as f:
            self.raw_points = np.loadtxt(f, delimiter=' ')
            self.raw_points = self.raw_points[:,1:]*scale + np.array([-0.3,-0.5])
        self.traj2linesegs()

        # self.line_segments = [((-1, -1), (-0.9, -1)), 
        #                       ((-0.9, -1), (-0.7, -0.9)), 
        #                       ((-0.7, -0.9), (-0.4, -0.8))]

    def traj2linesegs(self):
        points_tuple = [tuple(point) for point in self.raw_points]
    
        # Create list of line segments as tuples
        self.line_segments = [(points_tuple[i], points_tuple[i + 1]) for i in range(len(self.raw_points) - 1)]


    def find_intersection_pt(self, point_semgents, start_pt, direction):
        A,B = np.array(point_semgents[0]), np.array(point_semgents[1])
        C   = np.array(start_pt)
        D   = np.array(direction)
        
        # Create the matrix and the vector for the system of equations
        matrix = np.array([D, -(B - A)]).T
        vector = A - C
        
        # Solve the system of equations to find t and s
        t, s = np.linalg.solve(matrix, vector)
        
        # Calculate the intersection point using either of the parametric equations
        intersection_point = C + t * D
    
        return intersection_point

    def line_sign(self, curr_pos, line_dir, point):
        curr2pt = np.array(point) - np.array(curr_pos)
        cproduct = np.cross(np.array([line_dir[0],line_dir[1],0]), np.array([curr2pt[0],curr2pt[1],0]))
        return np.sign(cproduct[2])
    
    def find_traj_target(self, current_position, heading_vector):
        lateral_heading = np.array([heading_vector[1], -heading_vector[0]])

        intersect_lst = []
        for i in range(len(self.line_segments)):
            curr_seg_st, curr_seg_ed = self.line_segments[i]
            sign_st = self.line_sign(current_position, lateral_heading, curr_seg_st)
            sign_ed = self.line_sign(current_position, lateral_heading, curr_seg_ed)
            if sign_ed != sign_st:
                intersect_pt = self.find_intersection_pt(self.line_segments[i], current_position, lateral_heading)
                dist_to_intersect = np.linalg.norm(intersect_pt-current_position)
                intersect_lst.append([intersect_pt, dist_to_intersect, self.line_segments[i]])

        if not intersect_lst:
            dist_to_pts = np.linalg.norm(self.raw_points-current_position,axis=1)
            idx = np.argmin(dist_to_pts)
            traj_target = self.raw_points[idx]
            dist_to_target = dist_to_pts[idx]
            side_of_road = self.line_sign(current_position, heading_vector, traj_target)
            return traj_target, dist_to_target*side_of_road*self.direction

        
        traj_target_idx = np.argmin([entry[1] for entry in intersect_lst]) # point corresponding to the min distance
        traj_target, dist_to_target, min_segment = intersect_lst[traj_target_idx]

        side_of_road = -self.line_sign(min_segment[0], np.array(min_segment[1])-np.array(min_segment[0]), current_position)
        return traj_target, dist_to_target*side_of_road*self.direction

    def quat_to_heading(self, current_quat):
        current_orient = R.from_quat(current_quat)        
        current_euler  = current_orient.as_euler("YZX")
        heading_vector = R.from_euler("XYZ", [0,current_euler[0],0]).as_matrix()
        heading_vector = heading_vector[[0,2],0] # [x,z] of nose pointing in mocap frame (2D)
        return heading_vector

    def pose_callback(self, data):
        # Y up definition, x forward, Z right
        current_position = (data.pose.position.x, data.pose.position.z)
        current_quat = (data.pose.orientation.x, data.pose.orientation.y,
                        data.pose.orientation.z, data.pose.orientation.w)

        heading_vector = self.quat_to_heading(current_quat)
        # print(heading_vector, current_quat)

        # Find the lookahead point
        # first, find the corresponding location on trajectory
        traj_target, error = self.find_traj_target(current_position, heading_vector)
        
        # If no more segments are left or no lookahead point is found, you might want to stop the robot.
        if traj_target is None:
            # Implement stopping logic here
            self.steer_pub.publish(128) # [0,255] 128 is center
            self.throttle_pub.publish(0.0) # 0 rpm
            return

        # Calculate error based on the lookahead point
        # error = self.calculate_error(current_position, heading_vector, traj_target)
        error_curr_t = time.time()

        delta_t = error_curr_t - self.error_prev_t
        
        self.error_dot    = (error - self.error_prev)/delta_t
        self.error_prev   = error
        self.error_prev_t = error_curr_t
        
        steer_ctrl      = self.Kp * error + self.Kd * self.error_dot
        # print(steer_ctrl)
        steer_ctrl = min(max(int(steer_ctrl) + 127, 0), 255)
        # print(steer_ctrl)

        # Publish control input as a steering command
        if not self.kill:
            self.steer_pub.publish(steer_ctrl)
            self.throttle_pub.publish(min(self.speed, (time.time()-self.start_time)*30))

        # Lap counter
        dist = np.linalg.norm(np.array(current_position)-self.raw_points[0,0:2])
        curr_t = time.time()
        if (dist < 0.1) and (curr_t - self.last_t > 2):
            self.lap = self.lap+1
            self.last_t = curr_t
            print('Lap Count:',self.lap)

    def ena_callback(self, data):
        self.enable = data.data > 0.5
        self.enable_time = data.header.stamp

    def shutdown_hook(self):
        print("Shutting down... Setting throttle and steer to zero.")
        self.kill = True
        self.throttle_pub.publish(0)
        self.steer_pub.publish(127)
        time.sleep(0.1)

if __name__ == '__main__':
    try:
        controller = PDController(Kp=500, Kd=50, scale=0.4, speed=200, direction="CCW")
        rospy.on_shutdown(controller.shutdown_hook)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
        
