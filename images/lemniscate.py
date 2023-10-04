import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# Initialize an empty plot for the point p(t) and the curve
point, = ax.plot([], [], 'ro')
curve, = ax.plot([], [], 'k-')

# Function to initialize the plot
def init():
    point.set_data([], [])
    curve.set_data([], [])
    return point, curve

# Function to update the plot in each animation frame
def update(frame):
    t = frame * 0.01 * (2 * np.pi)  # Adjust the speed of animation here
    x = np.sin(t)
    y = np.cos(t) * np.sin(t)
    point.set_data(x, y)
    
    # Compute the curve up to the current t
    t_values = np.linspace(0, t, 1000)
    x_curve = np.sin(t_values)
    y_curve = np.cos(t_values) * np.sin(t_values)
    curve.set_data(x_curve, y_curve)
    
    return point, curve

# Create the animation
ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True)

# Save the animation as a GIF
ani.save('curve1_animation.gif', writer='pillow', fps=30)

# Show the animation
plt.show()



# Create a figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# Initialize an empty plot for the point p(t) and the curve
point, = ax.plot([], [], 'ro')
curve, = ax.plot([], [], 'k-')

# Function to initialize the plot
def init():
    point.set_data([], [])
    curve.set_data([], [])
    return point, curve

# Function to update the plot in each animation frame
def update(frame):
    t = frame * 0.01 * (2 * np.pi)  # Adjust the speed of animation here
    x = np.sin(2*t)
    y = np.cos(2*t) * np.sin(2*t)
    point.set_data(x, y)
    
    # Compute the curve up to the current t
    t_values = np.linspace(0, t, 1000)
    x_curve = np.sin(2*t_values)
    y_curve = np.cos(2*t_values) * np.sin(2*t_values)
    curve.set_data(x_curve, y_curve)
    
    return point, curve

# Create the animation
ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True)

# Save the animation as a GIF
ani.save('curve2_animation.gif', writer='pillow', fps=30)

# Show the animation
plt.show()
