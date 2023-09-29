# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| fig-cap: "Plot of Circle and Tangent Vector at $(0,1)$" 
#| label: fig-circle-tangent

import numpy as np
import matplotlib.pyplot as plt

# Create values for t ranging from 0 to 2*pi
t = np.linspace(0, 2 * np.pi, 100)

# Calculate x and y coordinates for the circle
x = np.cos(t)
y = np.sin(t)

# Plot the circle
plt.figure(figsize=(6, 6))
plt.plot(x, y, color = 'deeppink',label='Circle')
plt.axis('equal')  # Equal aspect ratio

# Define the point on the circle where you want to find the tangent
point_x = 0
point_y = 1

# Plot the point
plt.scatter(point_x, point_y, color='black', label='Point (0,1)')

# Calculate the tangent vector at the point (0,1)
tangent_vector_x = -np.sin(np.pi/2)  # Derivative of cos(t) with respect to t at t=pi/2
tangent_vector_y = np.cos(np.pi/2)   # Derivative of sin(t) with respect to t at t=pi/2

# Plot the tangent vector at the point (0,1)
plt.quiver(point_x, point_y, tangent_vector_x, tangent_vector_y,angles='xy', scale_units='xy', scale=1, color='blue', label='Tangent Vector')


# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Show the plot
plt.grid(color='lightgray')
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| fig-cap: "Plot of Helix with tangent vector" 
#| label: fig-helix-tangent

# Plotting 3D Helix

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Generates figure and 3D axes
fig = plt.figure(figsize = (4,4))
ax = plt.axes(projection = '3d')

# Plots grid
ax.grid(True)

# Divides time interval (0,6pi) in 100 parts 
t = np.linspace(-2*np.pi, 2*np.pi, 100)

# Computes Helix
x = np.cos(t) 
y = np.sin(t)
z = t

# Plots Helix - We added some styling
ax.plot3D(x, y, z, color = "deeppink", linewidth = 2)


# Point of interest (t = π/2)
t_interest = np.pi / 2
x_interest = np.cos(t_interest)
y_interest = np.sin(t_interest)
z_interest = t_interest

# Calculate the tangent vector at t = π/2
tangent_vector = np.array([-np.sin(t_interest), np.cos(t_interest), 1])

# Scale the tangent vector for visualization
scale_factor = 1
tangent_vector *= scale_factor

# Create a line representing the tangent vector at t = π/2
ax.quiver(x_interest, y_interest, z_interest,
          tangent_vector[0], tangent_vector[1], tangent_vector[2],
          color='blue', label='Tangent Vector', linewidth=2)

# Shows the plot
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| fig-cap: "A self-intersecting curve" 
#| label: fig-eight

import numpy as np
import matplotlib.pyplot as plt

# Create values for t ranging from 0 to 2*pi
t = np.linspace(0, 2 * np.pi, 100)

# Calculate x and y coordinates for the circle
x = np.sin(t)
y = np.cos(t)*np.sin(t)

# Plot the circle
plt.figure(figsize=(6, 6))
plt.plot(x, y, color = 'deeppink',label='Curve')
plt.axis('equal')  # Equal aspect ratio

# Define the point on the circle where you want to find the tangent
point_x = 0
point_y = 0

# Plot the point
plt.scatter(point_x, point_y, color='black', label='Point (0,0)')

# Calculate the tangent vectors at the point (0,0)
tangent_vector_x_1 = 1
tangent_vector_y_1 = 1

tangent_vector_x_2 = -1
tangent_vector_y_2 = 1

# Plot the tangent vectors at the point (0,0)
plt.quiver(point_x, point_y, tangent_vector_x_1, tangent_vector_y_1,angles='xy', scale_units='xy', scale=1, color='blue', label='Tangent Vector for $t=0$')
plt.quiver(point_x, point_y, tangent_vector_x_2, tangent_vector_y_2,angles='xy', scale_units='xy', scale=1, color='blue', label='Tangent Vector for $t = \pi$')


# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')

# Show the plot
plt.grid(color='lightgray')
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| fig-cap: "Plot of Logarithmic Spiral$" 
#| label: fig-log-spiral

import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the logarithmic spiral
a = 1  # Scaling factor
b = 1  # Growth factor

# Generate values for theta (angle)
theta = np.linspace(0, 10 * np.pi, 1000)

# Calculate the corresponding values of r (radius)
r = a * np.exp(b * theta)

# Convert polar coordinates to Cartesian coordinates
x = r * np.cos(theta)
y = r * np.sin(theta)

# Plot the logarithmic spiral
plt.figure(figsize=(8, 8))
plt.plot(x, y, label='Logarithmic Spiral', linewidth=2)
plt.title('Logarithmic Spiral')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.axis('equal')  # Equal scaling for X and Y axes
plt.show()
#
#
#
