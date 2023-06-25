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
        z[i, j] = -50 * np.exp(-((x[i, j] - 500)**2 + (y[i, j] - 500)**2) / 40000) + 50

# Create the trajectory array
trajectory = np.zeros((size,))
trajectory[0] = z[int(size/2), 0] + 30

trajectory_y = np.zeros((size,))
trajectory_y[0] = y[int(size/2), 0]
for i in range(1, size):
    # Calculate the x value for the current point on the trajectory
    x_val = x[0, i]
    # Calculate the y value for the current point on the trajectory
    y_val = y[int(size/2), i]
    # Interpolate the z value for the current point on the trajectory
    z_val = np.interp(x_val, x[0], z[:, i])
    # Calculate the height offset for the current point on the trajectory
    y_offset = 30 * np.sin(i/40)
    # Add the height offset to the interpolated z value
    trajectory[i] = z_val + 30
    trajectory_y[i] = y_val #+ y_offset

# Create a 3D surface plot
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, cmap='terrain', alpha=0.8)

# Add the trajectory to the plot
ax.plot(x[0], trajectory_y, trajectory, color='red')

# Add a color bar
fig.colorbar(surf)

# Set the axis labels
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Height (m)')

# Set the plot limits
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_zlim(0, 100)

# Show the plot
plt.show()
