#!/home/kang/anaconda3/envs/racecar/bin/python
import rospy
from std_msgs.msg import Float32
import asyncio
import numpy as np
from evdev import InputDevice, categorize, ecodes, ff

joy_states = {'steer': 128.0, 'throttle': 0.0, 'brake': 0.0}

# Create a ROS publisher
spub = rospy.Publisher('steer', Float32, queue_size=1)
tpub = rospy.Publisher('throttle', Float32, queue_size=1)

# Initialize the ROS node
rospy.init_node('G29_controller', anonymous=True)

# # Replace 'eventX' with the event number of your PS4 controller
# port = "/dev/input/by-id/usb-Logitech_G29_Driving_Force_Racing_Wheel-event-joystick"
# device = InputDevice(port)

# print(device)
# print(device.capabilities())

# if ecodes.EV_FF in device.capabilities():
#     # Set the autocentering strength
#     autocenter_strength = 0x4000  # Set this to the desired strength (0x0000 to 0xFFFF)
#     device.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, autocenter_strength)
#     print('Autocentering strength set!')

# else:
#     print('The device does not support force feedback.')

def clamp(vmin,val,vmax):
    return min(max(vmin,val),vmax)

# # Event handling function
# async def print_events(device):
#     global joy_states
#     try:
#         async for event in device.async_read_loop():
#             if event.type == ecodes.EV_KEY:
#                 c_event = categorize(event)
#                 # joy_msg.buttons.append(c_event.scancode)
#                 # joy_msg.buttons.append(c_event.keystate)
#             elif event.type == ecodes.EV_ABS:
#                 absevent = categorize(event)
#                 # print(ecodes.bytype[absevent.event.type][absevent.event.code], absevent.event.value)
#                 # joy_msg.axes.append(absevent.event.value)
#                 if absevent.event.code == 5:
#                     # print(absevent.event.value)
#                     joy_states['brake'] = 255-absevent.event.value
#                 if absevent.event.code == 2:
#                     # print(absevent.event.value)
#                     joy_states['throttle'] = 255-absevent.event.value
#                 if absevent.event.code == 0:
#                     joy_states['steer'] = absevent.event.value/65535*255
#                     joy_states['steer'] = (joy_states['steer']-255/2)*2+255/2 # scale by 2x
#     except KeyboardInterrupt:
#         print("Closing...")
#         return

async def print_events():
    print(joy_states)

# async def publish_to_ros1():
#     global joy_states
#     # Start event loop
#     while not rospy.is_shutdown():
#         steer = joy_states['steer']
#         throttle = joy_states['throttle']
#         if throttle < 3 and joy_states['brake'] > 3:
#             throttle = -joy_states['brake']

#         spub.publish(Float32(float(clamp(0,steer,255))))
#         tpub.publish(Float32(float(clamp(-255,throttle,255))))
#         # print('pub')
#         await asyncio.sleep(0.02)

async def publish_to_ros():
    
    # Start event loop
    while not rospy.is_shutdown():
        print('pub')
        steer = np.sin(np.pi*rospy.Time.now().to_sec())*20+128
        throttle = np.sin(np.pi*rospy.Time.now().to_sec())*100


        spub.publish(Float32(float(clamp(0,steer,255))))
        tpub.publish(Float32(float(clamp(-255,throttle,255))))
        # print('pub')
        await asyncio.sleep(0.02)

# Event loop for asynchronous reading of input events
loop = asyncio.get_event_loop()
task1 = loop.create_task(print_events())
task2 = loop.create_task(publish_to_ros())
loop.run_until_complete(asyncio.gather(task1, task2))