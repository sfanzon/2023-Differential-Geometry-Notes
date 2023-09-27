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
#| echo: false
#| fig-cap: "Plotting straight line $y=2x+1$" 
#| code-overflow: wrap


import numpy as np
import matplotlib.pyplot as plt

# Computing Spiral
t = np.linspace(-1, 1, 500)
x = t
y = 2*t + 1

# Generating figure
plt.figure(1, figsize = (4,4))

# Plotting the Spiral with some options
plt.plot(x, y, color = 'deeppink', linewidth = 1.5, label = 'y = 2x +1')

# Adding grid
plt.grid(True, color = 'lightgray')

# Adding axes labels
plt.xlabel("x-axis", fontsize = 15)
plt.ylabel("y-axis", fontsize = 15)

# Show the plot
plt.show()
#
#
#
#
#| echo: false
#| fig-cap: "Plot of hyperbole $y=e^x$" 
#| code-overflow: wrap


import numpy as np
import matplotlib.pyplot as plt

# Computing Spiral
t = np.linspace(-1, 1, 500)
x = t
y = np.exp(t)

# Generating figure
plt.figure(1, figsize = (4,4))

# Plotting the Spiral with some options
plt.plot(x, y, color = 'deeppink', linewidth = 1.5, label = 'y = 2x +1')


# Adding axes labels
plt.xlabel("x-axis", fontsize = 15)
plt.ylabel("y-axis", fontsize = 15)

plt.axvline(x=0, c="gray", label="x=0")
plt.axhline(y=0, c="gray", label="y=0")


# Show the plot
plt.show()
#
#
#
#
#
#| echo: false
#| fig-cap: "Plot of unit circle of equation $x^2 + y^2 = 1$" 

# Plotting f=0

import numpy as np
import matplotlib.pyplot as plt

# Generates coordinates and grid
xlist = np.linspace(-2, 2, 5000)
ylist = np.linspace(-2, 2, 5000)
X, Y = np.meshgrid(xlist, ylist)

# Computes f
Z = X**2 + Y**2

# Creates figure object
plt.figure(figsize = (4,4))

# Plots level set Z = 0
plt.contour(X, Y, Z, [1])

# Set axes labels
plt.xlabel("x-axis", fontsize = 15)
plt.ylabel("y-axis", fontsize = 15)

# Shows plot
plt.show()
#
#
#
#
#
