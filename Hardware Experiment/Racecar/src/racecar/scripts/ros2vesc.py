#!/home/racecar/mambaforge/envs/racecar/bin/python3
import serial,time
from pyvesc import VESC
import rospy
from std_msgs.msg import Float32
import numpy as np
import signal

# ===========Killer=========
def signal_handler(signal, frame):
    global loop_running
    loop_running = False
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
loop_running = True
# ==========================

def lipo_volt2percent(voltage):
    if voltage < 9.82:
        return -1
    # Typical voltage levels for a single-cell LiPo
    voltage_points = [9.82, 10.83, 11.06, 11.12, 11.18,
                      11.24, 11.3, 11.36, 11.39, 11.45,
                      11.51, 11.56, 11.62, 11.74, 11.86, 
                      11.95, 12.07, 12.25, 12.33, 12.45, 12.6]
    # percentage_points = [0, 10, 20, 40, 60, 80, 100]
    percentage_points = list(range(0,105,5))

    # Find the two points in the curve that the voltage lies between
    for i in range(1, len(voltage_points)):
        if voltage < voltage_points[i]:
            # Linearly interpolate between the two points
            percentage = percentage_points[i-1] + (percentage_points[i] - percentage_points[i-1]) * (voltage - voltage_points[i-1]) / (voltage_points[i] - voltage_points[i-1])
            return percentage

    # If voltage is greater than the maximum voltage point, return 100
    return 100.0

throttle = 0.0

def throttle_callback(data):
    global throttle
    global rpm
    throttle = (data.data/255)*0.3
    if abs(rpm) < 1500:
        throttle = np.sign(throttle)* min(2500/20000, abs(throttle))

serial_port = '/dev/serial/by-id/usb-STMicroelectronics_ChibiOS_RT_Virtual_COM_Port_304-if00'

vesc = VESC(serial_port)

rospy.init_node('VESCSerial', anonymous=True)
rospy.Subscriber("throttle", Float32, throttle_callback)
rate = rospy.Rate(50) # 10 Hz

v_in_pub  = rospy.Publisher('V_in', Float32, queue_size=1)
rpm_pub   = rospy.Publisher('RPM', Float32, queue_size=1)
duty_pub  = rospy.Publisher('Duty_cycle', Float32, queue_size=1)
tach_pub  = rospy.Publisher('Tach', Float32, queue_size=1)
power_pub = rospy.Publisher('P_motor', Float32, queue_size=1)
batt_pub  = rospy.Publisher('Batt', Float32, queue_size=1)
speed_pub = rospy.Publisher('Wheel_Speed', Float32, queue_size=1)

# averages
vin_lst = []

count = 0
rpm = 0
while loop_running:
    try:
        if not (throttle is None):
            # print("Throttle ",throttle)
            if abs(throttle) < 0.04:
                vesc.set_current(0.0)
            else:
                if throttle < 0:
                    vesc.set_rpm(int(throttle*7000))
                else:
                    vesc.set_rpm(int(throttle*15000)) # minimum is 588erpm, 9.8rps

        if count%1 == 0:
            status = vesc.get_measurements()
            if status is not None:
                # update values
                vin_lst.append(status.v_in)
                rpm = status.rpm

                batt_percent = lipo_volt2percent(np.mean(vin_lst))

                # publish values
                v_in_pub.publish(np.mean(vin_lst))
                batt_pub.publish(batt_percent)
                rpm_pub.publish(status.rpm)
                duty_pub.publish(status.duty_cycle_now)
                tach_pub.publish(status.tachometer)
                power_pub.publish(status.avg_input_current*status.v_in)
                speed = status.rpm/60/10 * 0.067 * 3.1415927
                speed_pub.publish(speed)

                # manage lists
                if len(vin_lst) > 400:
                    vin_lst.pop(0)
                
                # print("Duty:",status.duty_cycle_now)
            else:
                # print('Status is none')
                pass

        count += 1
        rate.sleep()

    except Exception as e:
        print(f"Killing for: {e}")