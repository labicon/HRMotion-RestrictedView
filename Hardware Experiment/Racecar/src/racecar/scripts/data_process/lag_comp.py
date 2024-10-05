# /usr/bin/env python
import rosbag
import yaml
import argparse
from tqdm import tqdm

class lag_compensator:
    def __init__(self, fpath):
        self.fpath      = fpath
        self.bag        = rosbag.Bag(self.fpath)
        self.info_dict  = yaml.safe_load(self.bag._get_yaml_info())
        print(self.info_dict.keys())

    def read_msg(self):
        # for each message, populate the training set array
        for topic, msg, t in self.bag.read_messages(): # topics=['/step_counter','/rosout']
            self.extract_data(topic,msg,t)

    def extract_data(self, topic, msg, t):
        print(topic, msg, t)

    def lag_compensate(self, outfile):
        print('Rewriting according to raw time stamp')
        with rosbag.Bag(outfile, 'w') as outbag:
            with tqdm(total=self.bag.get_message_count()) as pbar:
                for topic, msg, t in self.bag.read_messages():
                    outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)
                    pbar.update(1)

    # Clean up function
    def exit(self):
        self.bag.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath',      type=str,   default='??')
    args = parser.parse_args()

    # fpath = '../bag_recording/field_2021-07-16-11-46-47.bag'
    # fpath = '../bag_recording/field_2021-08-18-15-59-25.bag'
    # fpath = '../bag_recording/field_2021-12-09-16-45-58.bag.active'
    # fpath = '/media/askker/Extreme Pro/bag_recording/field_2021-12-09-16-53-24.bag'
    # outfile = '/media/askker/Extreme Pro/bag_recording/field_2021-12-09-16-53-24lag.bag'
    fpath   = args.fpath
    outfile = ''.join(args.fpath.split('.')[:-1]+['_lag.bag'])

    lag = lag_compensator(fpath)
    # reader.read_msg()
    lag.lag_compensate(outfile)

    # Clean up
    lag.exit()