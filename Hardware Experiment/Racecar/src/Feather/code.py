# import board
# import busio
# import digitalio
# import supervisor
# import time
# from adafruit_servokit import ServoKit

# def clamp(vmin,val,vmax):
#     return max(min(val,vmax),vmin)

# # Initialize
# kit = ServoKit(channels=8)
# kit.servo[4].angle = 85 # 30-85-140
# print("listening...")
# t1 = time.monotonic()

# # Setup serial loop
# while True:
#     # ts = time.monotonic()
#     if supervisor.runtime.serial_bytes_available:
#         data = input().strip()
#         if data is not None:
#             servo_angle = clamp(30,float(data),140)
#             kit.servo[4].angle = servo_angle
#             print("RCVD:"+data)
#             t1 = time.monotonic()
    
#     # in case of connection lost
#     if time.monotonic()-t1 > 0.5:
#         kit.servo[4].angle = 85

#     # te = time.monotonic()
#     # print('Time:',te-ts)


import busio
import supervisor
import time
from board import SCL, SDA
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685


def clamp(vmin,val,vmax):
    return max(min(val,vmax),vmin)

# Initialize
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50
kit = servo.Servo(pca.channels[4])
print("listening...")
t1 = time.monotonic()

# Setup serial loop
while True:
    # ts = time.monotonic()
    if supervisor.runtime.serial_bytes_available:
        data = input().strip()
        if data is not None:
            servo_angle = clamp(30,float(data),140)
            kit.angle = servo_angle
            # print("RCVD:"+data)
            t1 = time.monotonic()
    
    # in case of connection lost
    if time.monotonic()-t1 > 0.5:
        kit.angle = 85

    # te = time.monotonic()
    # print('Time:',te-ts)
