import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# Define the terrain parameters
size = 100  # Size of the terrain (number of facets)
scale = 500  # Scale of the terrain (maximum height difference)
height_map = np.zeros((size, size))  # Empty height map

# Add changes in height using sine and cosine functions
for i in range(size):
    for j in range(size):
        height = np.sin(i/10) * np.cos(j/10) * scale
        # Limit the height to be between 0 and 200 meters
        height = max(0, height)
        height = min(200, height)
        height_map[i,j] = height

# Apply a Gaussian smoothing filter
sigma = 10  # Standard deviation of the filter
height_map = gaussian_filter(height_map, sigma=sigma)

# Create a 3D surface plot
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,0.5])

x, y = np.meshgrid(np.arange(size), np.arange(size))
print(x)
surf = ax.plot_surface(x, y, height_map, cmap='terrain', alpha=0.8)

# Add a color bar
fig.colorbar(surf)
# Set the axis labels
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Height (m)')
# Set the tick intervals
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 110, 10))
# Show the plot
plt.show()
