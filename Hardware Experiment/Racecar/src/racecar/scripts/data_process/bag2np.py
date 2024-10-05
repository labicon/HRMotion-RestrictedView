# /usr/bin/env python
import rosbag
import yaml
import argparse
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np
import pickle

class bag_converter:
    def __init__(self, fpath, cut=0.0):
        print('Cutting away last:', cut, 'secs')
        self.fpath      = fpath
        self.cut        = cut
        self.bag        = rosbag.Bag(self.fpath)
        self.info_dict  = yaml.safe_load(self.bag._get_yaml_info())
        self.data_array = []
        self.dataline = [0,0,0,1,0,0,0,0] # (8,)
        print(self.info_dict.keys())

    def read_msg(self):
        # for each message, populate the training set array
        with tqdm(total=self.bag.get_message_count()) as pbar:
            for topic, msg, t in self.bag.read_messages(): # topics=['/step_counter','/rosout']
                old_dataline = self.dataline.copy()
                self.extract_data(topic,msg,t)
                if old_dataline != self.dataline:
                    self.data_array.append(self.dataline.copy())
                pbar.update(1)
        self.data_array = np.array(self.data_array[1:])

    def quat_to_heading(self, current_quat):
        current_orient = R.from_quat(current_quat)        
        current_euler  = current_orient.as_euler("YZX")
        heading_vector = R.from_euler("XYZ", [0,current_euler[0],0]).as_matrix()
        heading_vector = heading_vector[[0,2],0] # [x,z] of nose pointing in mocap frame (2D)
        return heading_vector

    def extract_data(self, topic, msg, t):
        if t.to_sec() < self.dataline[0]:
            print("lag comp is not working")

        if topic=='/vrpn_client_node/Racecar/pose':
            self.dataline[0] = t.to_sec()
            self.dataline[1] = msg.pose.position.x
            self.dataline[2] = msg.pose.position.z

            quat    = [msg.pose.orientation.x, msg.pose.orientation.y, 
                       msg.pose.orientation.z, msg.pose.orientation.w]
            heading = self.quat_to_heading(quat)
            self.dataline[3] = heading[0]
            self.dataline[4] = heading[1]

        if topic=='/Wheel_Speed':
            self.dataline[0] = t.to_sec()
            self.dataline[5] = msg.data

        if topic=='/steer':
            self.dataline[0] = t.to_sec()
            self.dataline[6] = msg.data
        
        if topic=='/throttle':
            self.dataline[0] = t.to_sec()
            self.dataline[7] = msg.data

    def resample(self, hz):
        print('Resampling to ',hz,"HZ")
        t_st = self.data_array[0,0]
        t_ed = self.data_array[-1,0]
        print("duration:",t_ed-t_st)
        resampled_data = []
        for t in tqdm(np.arange(t_st, t_ed-self.cut, 1/hz)):
            array_idx = np.argmin(abs(self.data_array[:,0]-t))
            resampled_data.append(self.data_array[array_idx])

        self.resampled_data = np.array(resampled_data)
        print(self.resampled_data.shape)

    def save(self):
        print('Saving...')

        save_fpath = self.fpath[:-4]+'_np'
        print(self.fpath)
        # Write to file
        outfile = open(save_fpath,'wb')           # Fastest 57s 4300MB
        # outfile = bz2.BZ2File(save_fpath, 'w')    # Least space 473s 341MB
        # outfile = gzip.open(save_fpath,'wb')      # most economical 310s 493MB
        pickle.dump(self.resampled_data, outfile)
        outfile.close()

        print('Done')

    # Clean up function
    def exit(self):
        self.bag.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath',      type=str,   default='??')
    parser.add_argument('--cut', type=float, default=0)
    args = parser.parse_args()
    fpath   = args.fpath
    cut = args.cut

    converter = bag_converter(fpath, cut)
    converter.read_msg()
    print(converter.data_array.shape)
    converter.resample(hz=20)
    converter.save()
    # converter.convert()

    # Clean up
    converter.exit()