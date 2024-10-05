#!/usr/bin/python3
import rospy
import cv2
import gi
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

def i420_to_bgr(array, width=1280, height=720):
    y_size = width * height
    u_size = v_size = y_size // 4

    # Create a single channel image where Y, U, and V planes are stacked
    yuv_stacked = array.reshape((int(height * 1.5), width))

    # Convert to BGR
    bgr_image = cv2.cvtColor(yuv_stacked, cv2.COLOR_YUV2BGR_I420)
    
    return bgr_image

def on_new_buffer(appsink):
    print('on_new_buffer')
    sample = appsink.emit('pull-sample')
    buffer = sample.get_buffer()
    # caps = sample.get_caps()
    # ff = buffer.extract_dup(0, buffer.get_size())
    array = buffer.extract_dup(0, buffer.get_size())
    
    array = np.frombuffer(array, dtype=np.uint8)
    # print("Array shape:", array.shape)
    # print("Array head:", array[:10])
    bgr_image = i420_to_bgr(array)
    # image = cv2.imdecode(array, 1)
    
    bridge = CvBridge()
    image_message = bridge.cv2_to_imgmsg(bgr_image, encoding="bgr8")
    
    pub.publish(image_message)
    
    return Gst.FlowReturn.OK

def main():
    global pub
    rospy.init_node('gstreamer_to_ros', anonymous=True)
    pub = rospy.Publisher('/FPV_video', Image, queue_size=2)

    Gst.init(None)
    pipeline_str = 'udpsrc port=5003 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink name=appsink sync=false'
    pipeline_str = 'udpsrc port=5003 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! avdec_h264 ! tee name=t ! queue ! videoconvert ! autovideosink t. ! queue ! videoconvert ! appsink name=appsink sync=false'
    pipeline = Gst.parse_launch(pipeline_str)
    print('parsed')
    appsink = pipeline.get_by_name('appsink')
    appsink.set_property("emit-signals", True)

    print('appsink')
    
    appsink.connect('new-sample', on_new_buffer)
    
    pipeline.set_state(Gst.State.PLAYING)
    rospy.spin()
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    main()
