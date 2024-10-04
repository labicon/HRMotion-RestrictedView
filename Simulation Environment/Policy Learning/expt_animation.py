# write a code to generate animation of bicycle plots

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

def create_obsts_fixed(x_positions, y_positions):
    radius = []; circ_x = []; circ_y = []
    print(len(x_positions))
    for i in range(len(x_positions)):
        radius.append(random.randint(10, 10))
        circ_x.append((x_positions[i])*420/1000)
        circ_y.append((y_positions[i])*360/600)
    return [radius, circ_x, circ_y]

# data
positions = np.loadtxt("data/bicycle_positions_expt_3.csv", delimiter=",")
# obstacles = np.loadtxt("./src/racecar/scripts/Remote/bicycle_diff_data/obstacles_3.csv", delimiter=",")

# obstacles[:,1] = obstacles[:,1] - 180
# obstacles[:,2] = obstacles[:,2] - 180

# print(obstacles)

# obstacles = np.loadtxt("./src/racecar/scripts/Remote/bicycle_diff_data/obstacles_3.csv", delimiter=",")

x_positions = [344.486, 483.842, 601.642, 871.514, 505.249, 606.248, 367.165]
y_positions = [508.149, 299.161, 118.530, 362.297, 478.287, 403.393, 306.231]
[radius, circ_x, circ_y] = create_obsts_fixed(x_positions, y_positions)

obstacles = np.array([radius, circ_x, circ_y]).T

# create figure
fig, ax = plt.subplots()
ax.scatter(obstacles[:, 1], obstacles[:, 2], c='b', s=10)
ax.scatter(415, 355, c='g', s=10)
ax.set_xlim(-10, 420)
ax.set_ylim(-10, 360)
ax.set_aspect('equal')
# ax.grid()

# plot
line, = ax.plot([], [], 'r-', lw=5)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function. This is called sequentially

def animate(i):
    x = positions[i, 0]
    y = positions[i, 1]
    theta = positions[i, 2]
    l = 38
    x1 = x + l/2 * np.cos(theta)
    y1 = y + l/2 * np.sin(theta)
    x2 = x - l/2 * np.cos(theta)
    y2 = y - l/2 * np.sin(theta)
    line.set_data([x1, x2], [y1, y2])
    return line,

# call the animator. blit=True means only re-draw the parts that have changed.
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=positions.shape[0], interval=500, blit=True)

#save the animation as a gif
# ani.save('./src/racecar/scripts/Remote/bicycle_diff_data/bicycle_diff.gif', writer='imagemagick', fps=1)
# ani.save('bicycle_diff_data/bicycle_diff_3.gif', writer='ffmpeg', fps=30)

plt.show()

