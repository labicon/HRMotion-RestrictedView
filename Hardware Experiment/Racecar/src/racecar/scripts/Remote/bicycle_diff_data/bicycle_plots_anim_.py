# write a code to generate animation of bicycle plots

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# data
positions = np.loadtxt("bicycle_diff_data/positions_3.csv", delimiter=",")
obstacles = np.loadtxt("bicycle_diff_data/obstacles_3.csv", delimiter=",")

# create figure
fig, ax = plt.subplots()
ax.scatter(obstacles[:, 1], obstacles[:, 2], c='b', s=10)
ax.scatter(415, 355, c='g', s=10)
ax.set_xlim(0, 420)
ax.set_ylim(0, 360)
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
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=positions.shape[0], interval=60, blit=True)

#save the animation as a gif
ani.save('bicycle_diff_data/bicycle_diff_3.gif', writer='imagemagick', fps=10)
# ani.save('bicycle_diff_data/bicycle_diff_3.gif', writer='ffmpeg', fps=30)

plt.show()

