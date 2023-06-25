import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the terrain parameters
size = 1000  # Size of the terrain (number of facets)
scale = 50  # Scale of the terrain (maximum height difference)

# Create the x and y arrays
x = np.linspace(0, 1000, size)
y = np.linspace(0, 1000, size)
x, y = np.meshgrid(x, y)

# Create the z array using the desired function
z = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        z[i, j] = -50 * np.exp(-((x[i, j] - 500)**2 + (y[i, j] - 500)**2) / 200000)+50

# Create a 3D surface plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, cmap='terrain', alpha=0.8)

# Add a color bar
fig.colorbar(surf)

# Set the axis labels
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Height (m)')

# Set the plot limits
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_zlim(0, 50)

# Set the tick intervals
ax.set_xticks(np.arange(0, 1001, 100))
ax.set_yticks(np.arange(0, 1001, 100))

# Show the plot
plt.show()
