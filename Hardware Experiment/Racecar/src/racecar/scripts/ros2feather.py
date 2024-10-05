#!/home/racecar/mambaforge/envs/racecar/bin/python3
import serial
import time
import rospy
from std_msgs.msg import Float32
import signal

# ===========Killer=========
def signal_handler(signal, frame):
    global loop_running
    loop_running = False
    # Close the serial port.
    ser.close()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
loop_running = True
# ==========================

def steer_callback(data):
    global steer_angle
    steer_angle = (data.data/255-0.5)*110+85

HZ = 50
rospy.init_node('FeatherSerial', anonymous=True)
rospy.Subscriber("steer", Float32, steer_callback)
rate = rospy.Rate(HZ)
steer_angle = 85.0

# Open the serial port.
port = '/dev/serial/by-id/usb-Adafruit_Feather_M4_CAN_A88A3C905339473237202020FF113638-if00'
ser = serial.Serial(port, 115200, timeout=0.01)
print('Connected')
time.sleep(2)

# Write a command to the Feather board.
ser.write(b"90\n")
t2 = time.time()
# ser.flush()
print('Sent command:')

# Read the response.
while loop_running:
    try:
        t1 = time.time()
        # print('\n[============]')
        response = ser.readline().decode()
        # print("RS:",response)
        while ser.in_waiting:
            response = ser.readline().decode()
            if response[0] == "R":
                print('Response:', response)
                # print(time.time()-t2)
                # print('dd',time.time()-t1)
        
        cmd = (str(int(steer_angle))+"\n").encode()
        # print(cmd)
        ser.write(cmd)
        ser.flush()
        # print('Done Writing')

        # t2 = time.time()
        # time.sleep(max(0,1/HZ-(t2-t1)-0.00006))
        # t3 = time.time()
        # print('Time:',t3-t1)
        rate.sleep()

    except Exception as e:
        print(f"Killing for: {e}")