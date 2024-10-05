#!/home/kang/anaconda3/envs/racecar/bin/python
import rospy
from std_msgs.msg import Float32
import asyncio
from evdev import InputDevice, categorize, ecodes

joy_states = {'steer': 0.0, 'throttle': 0.0}

# Create a ROS publisher
spub = rospy.Publisher('steer', Float32, queue_size=1)
tpub = rospy.Publisher('throttle', Float32, queue_size=1)

# Initialize the ROS node
rospy.init_node('ps4_controller', anonymous=True)

# Replace 'eventX' with the event number of your PS4 controller
port = "/dev/input/by-id/usb-Sony_Interactive_Entertainment_Wireless_Controller-if03-event-joystick"
device = InputDevice(port)

print(device)

# Event handling function
async def print_events(device):
    global joy_states
    try:
        async for event in device.async_read_loop():
            if event.type == ecodes.EV_KEY:
                c_event = categorize(event)
                # joy_msg.buttons.append(c_event.scancode)
                # joy_msg.buttons.append(c_event.keystate)
            elif event.type == ecodes.EV_ABS:
                absevent = categorize(event)
                # print(ecodes.bytype[absevent.event.type][absevent.event.code], absevent.event.value)
                # joy_msg.axes.append(absevent.event.value)
                if absevent.event.code == 3:
                    # print(absevent.event.value)
                    joy_states['steer'] = absevent.event.value
                if absevent.event.code == 1:
                    # print(absevent.event.value)
                    joy_states['throttle'] = absevent.event.value
    except KeyboardInterrupt:
        print("Closing...")
        return

async def publish_to_ros():
    global joy_states
    # Start event loop
    while not rospy.is_shutdown():
        steer = joy_states['steer']
        throttle = joy_states['throttle']
        spub.publish(Float32(float(steer)))
        tpub.publish(Float32(float(throttle)))
        await asyncio.sleep(0.02)

# Event loop for asynchronous reading of input events
loop = asyncio.get_event_loop()
task1 = loop.create_task(print_events(device))
task2 = loop.create_task(publish_to_ros())
loop.run_until_complete(asyncio.gather(task1, task2))