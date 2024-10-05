import rospy
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import numpy as np
import threading, time
from scipy.spatial.transform import Rotation as R

# Global variable to store the last 100 poses
pose_list = []
folder_path = '/home/racecar/Documents/racecar_ws/src/Racecar/src/racecar/scripts/trajs/'
# traj_name = 'star500.txt'
# traj_name = 'circle.txt'
# traj_name = 'kinsmen500.txt'
# traj_name = 'buttonwillow1000.txt'
# traj_name = 'lagunaseca1000.txt'
traj_name = 'validation1000.txt'
with open(folder_path+traj_name, 'rb') as f:
    traj = np.loadtxt(f, delimiter=' ')
    traj = traj[:,1:]*1.5 + np.array([-0.0,-0.5])

def quat_to_heading(current_quat):
    current_orient = R.from_quat(current_quat)        
    current_euler  = current_orient.as_euler("YZX")
    heading_vector = R.from_euler("XYZ", [0,current_euler[0],0]).as_matrix()
    heading_vector = heading_vector[[0,2],0] # [x,z] of nose pointing in mocap frame (2D)
    return heading_vector
    
def pose_callback(pose_msg):
    global pose_list
    global new_pose
    global heading_vect
    global current_pose

    # Append new pose to list
    new_pose = [-pose_msg.pose.position.x, pose_msg.pose.position.z]
    current_quat = (pose_msg.pose.orientation.x, pose_msg.pose.orientation.y,
                    pose_msg.pose.orientation.z, pose_msg.pose.orientation.w)
    
    heading_vec = quat_to_heading(current_quat)
    # print(new_pose, heading_vec*0.275)
    heading_vect = np.array([-heading_vec[0], heading_vec[1]])
    current_pose = new_pose.copy()
    # new_pose = new_pose - heading_vect*0.275
    pose_list.append(new_pose)
    # print(new_pose)

    # Keep only the last 100 poses
    pose_list = pose_list[-300:]
    time.sleep(0.1)

def control():
    traj_points = traj
    L = 0.275
    current_pos = np.array([-current_pose[0],current_pose[1]])
    heading_vec = np.array([-heading_vect[0],heading_vect[1]])
    lookahead_distance = 0.5
    # print(current_pos, heading_vec)

    current_tail_pos = current_pos - heading_vec*L
    # Find the closest point to the vehicle on the trajectory
    closest_idx = np.argmin(np.linalg.norm(traj_points - current_tail_pos, axis=1))
    closest_pt = traj_points[closest_idx]
    # Search for the lookahead target point along the trajectory
    for i in range(closest_idx, len(traj_points)):
        if np.linalg.norm(traj_points[i] - current_tail_pos) >= lookahead_distance:
            target_point = traj_points[i]
            break

    # Calculate the steering angle based on the target point
    # pure pursuit
    
    tail_to_la_vec = target_point-current_tail_pos
    dist_la = np.linalg.norm(tail_to_la_vec)
    alpha = np.arccos(np.dot(tail_to_la_vec/dist_la, heading_vec))
    alpha = alpha*np.sign(np.cross(heading_vec,tail_to_la_vec))
    steer_angle = np.arctan2(2*L*np.sin(alpha),dist_la)
    print(target_point)
    print(np.rad2deg(steer_angle))
    # steer_angle = np.arctan2(target_point[1] - current_pos[1], target_point[0] - current_pos[0])

    # Publish the steering angle
    steer_command = int(np.rad2deg(steer_angle)/25*128+127)
    plt.scatter(-target_point[0],target_point[1],100,c='k',marker='x')
    plt.scatter(-closest_pt[0],closest_pt[1],100,c='k',marker='o')

def plot_trajectory():
    global pose_list
    global traj

    plt.figure()
    while not rospy.is_shutdown():
        # Convert to NumPy array for easier slicing
        local_pose_list = list(pose_list)  # Create a local copy to avoid race conditions
        pose_array = np.array(local_pose_list)

        # Plotting
        plt.clf()  # Clear the previous plot
        plt.plot(-traj[:,0],traj[:,1],'r')
        plt.plot(pose_array[:, 0], pose_array[:, 1], 'b-')
        current_tail_pos = pose_list[-1] - heading_vect*0.275
        plt.arrow(current_tail_pos[0], current_tail_pos[1], heading_vect[0]*0.275, heading_vect[1]*0.275, width=0.1, head_width=0.2, head_length=0.1)
        # control()
        plt.axis('equal')
        plt.xlim((-2,2))
        plt.ylim((-2.5,1.5))
        plt.grid()
        plt.pause(0.1)  # Pause for a short period to allow the plot to update
        

if __name__ == '__main__':
    rospy.init_node('pose_plotter', anonymous=True)

    # Create a separate thread for plotting
    plot_thread = threading.Thread(target=plot_trajectory)
    plot_thread.start()

    # Subscribe to the pose topic
    rospy.Subscriber("/vrpn_client_node/Racecar/pose", PoseStamped, pose_callback, queue_size=1)

    # Keep the node running
    rospy.spin()
