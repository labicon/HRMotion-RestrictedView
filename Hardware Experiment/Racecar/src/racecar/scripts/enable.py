#!/home/racecar/mambaforge/envs/racecar/bin/python3
import rospy
from std_msgs.msg import Float32
import signal

# ===========Killer=========
def signal_handler(signal, frame):
    global loop_running
    loop_running = False
    # Close the serial port.
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
loop_running = True
# ==========================


rospy.init_node('enabler', anonymous=True)
ena_pub = rospy.Publisher('enable', Float32, queue_size=1)
rate = rospy.Rate(10)

loop_running = True
while loop_running:
    try:
        ena_pub.publish(1.0)
        rate.sleep()
    except Exception as e:
        print(f"Killing for: {e}")