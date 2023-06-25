import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d

from scipy.interpolate import RegularGridInterpolator

# Параметры местности
size = 1000  # Size of the terrain (number of facets)
scale = 50  # Scale of the terrain (maximum height difference)

# Инициализация массивов координат
x = np.linspace(0, 1000, size)
y = np.linspace(0, 1000, size)
x, y = np.meshgrid(x, y)

# Создание траектории по оси Z высот
z = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        z[i, j] = -50 * np.exp(-((x[i, j] - 500)**2 + (y[i, j] - 500)**2) / 40000) + 50

# Инициализация траектории и ее начальной точки
trajectory_z = np.zeros((size,))
trajectory_z[0] = z[int(size/2), 0] + 30

trajectory_y = np.zeros((size,))
trajectory_y[0] = y[int(size/2), 0]

trajectory_x = np.zeros((size,))
trajectory_x[0] = 0
for i in range(1, size):
    # Calculate the x value for the current point on the trajectory
    x_val = x[0, i]
    # Calculate the y value for the current point on the trajectory
    y_val = y[int(size/2), i]
    # Interpolate the z value for the current point on the trajectory
    z_val = np.interp(x_val, x[0], z[:, i])
    # Calculate the height offset for the current point on the trajectory
    y_offset = 30 * np.sin(i/60)
    # Add the height offset to the interpolated z value
    trajectory_z[i] = z_val + 30
    trajectory_y[i] = y_val + y_offset
    trajectory_x[i] = x_val

# График рельефа и траектории
""" fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, cmap='terrain', alpha=0.8)
ax.plot(trajectory_x, trajectory_y, trajectory_z, color='red')
fig.colorbar(surf)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Height (m)')
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_zlim(0, 100)
plt.show() """

# Инициализация массива углов тангажа
pitch_angle_deg = np.zeros(size)

# Заполнение массива углов тангажа
for i in range(size):
    if 0 <= i < 10:
        pitch_angle_deg[i] = 10
    elif 200 <= i < 500:
        pitch_angle_deg[i] = -40
    elif 500 <= i < 800:
        pitch_angle_deg[i] = 40
    elif 800 <= i:
        pitch_angle_deg[i] = 0
        
pitch_angle_deg = gaussian_filter(pitch_angle_deg, sigma=50)

row_angle_deg = np.zeros(size)

for i in range(size):
    if 0 <= i < 30:
        row_angle_deg[i] = 20
    elif 30 <= i < 100:
        row_angle_deg[i] = 0
    elif 100 <= i < 130:
        row_angle_deg[i] = -20
    elif 130 <= i < 200:
        row_angle_deg[i] = 0
    elif 200 <= i < 230:
        row_angle_deg[i] = 20
    elif 230 <= i < 300:
        row_angle_deg[i] = 0
    elif 300 <= i < 330:
        row_angle_deg[i] = 20
    elif 330 <= i < 400:
        row_angle_deg[i] = 0
    elif 400 <= i < 430:
        row_angle_deg[i] = -20
    elif 430 <= i < 500:
        row_angle_deg[i] = 0
    elif 500 <= i < 530:
        row_angle_deg[i] = 20
    elif 530 <= i < 600:
        row_angle_deg[i] = 0
    elif 630 <= i < 700:
        row_angle_deg[i] = 20
    elif 530 <= i < 600:
        row_angle_deg[i] = 0
    elif 700 <= i < 730:
        row_angle_deg[i] = -20
    elif 730 <= i:
        row_angle_deg[i] = 0
        
row_angle_deg = gaussian_filter(row_angle_deg, sigma=10)
""" # График тангажа
plt.figure()
plt.plot(trajectory_x, pitch_angle_deg)
plt.xlabel("Distance")
plt.ylabel("Pitch angle (degrees)")
plt.title("Pitch angle vs Distance")
plt.show()
# График крена
plt.figure()
plt.plot(trajectory_x, row_angle_deg)
plt.xlabel("Distance")
plt.ylabel("Pitch angle (degrees)")
plt.title("Pitch angle vs Distance")
plt.show() """
def beam_intersection(trajectory_point, pitch_angle_rad, roll_angle_rad, angle_offset, search_step):
    x_val, y_val, z_val = trajectory_point
    
    projected_x = x_val + search_step * (np.cos(pitch_angle_rad) * np.cos(roll_angle_rad) - np.sin(angle_offset) * np.sin(pitch_angle_rad) * np.sin(roll_angle_rad))
    projected_y = y_val + search_step * (np.cos(pitch_angle_rad) * np.sin(roll_angle_rad) + np.sin(angle_offset) * np.sin(pitch_angle_rad) * np.cos(roll_angle_rad))
    projected_z = z_val - search_step * np.cos(angle_offset) * np.sin(pitch_angle_rad)

    return projected_x, projected_y, projected_z


def calculate_beam_length(trajectory_x, trajectory_y, trajectory_z, terrain_x, terrain_y, terrain_z, pitch_angle_deg, roll_angle_deg, angle_between_beams_deg=30, search_range=100):
    # Convert pitch, roll, and angle between beams from degrees to radians
    pitch_angle_rad = np.radians(pitch_angle_deg)
    roll_angle_rad = np.radians(roll_angle_deg)
    angle_between_beams_rad = np.radians(angle_between_beams_deg)

    # Initialize empty arrays for the beam lengths
    beam_length_1 = np.zeros(trajectory_x.shape)
    beam_length_2 = np.zeros(trajectory_x.shape)

    # Calculate the beam length for each point in the trajectory
    for i in range(trajectory_x.shape[0]):
        # Get the x, y, and z coordinates of the current trajectory point
        x_val = trajectory_x[i]
        y_val = trajectory_y[i]
        z_val = trajectory_z[i]

        min_distance_1 = float('inf')
        min_distance_2 = float('inf')

        for search_step in range(-search_range, search_range + 1):
            # Calculate the terrain point coordinates based on the projection of the beams considering the pitch and roll angles and angle between the beams
            for beam_idx, angle_offset in enumerate([-angle_between_beams_rad / 2, angle_between_beams_rad / 2]):
                roll_offset = roll_angle_rad[i] if beam_idx == 0 else -roll_angle_rad[i]
                projected_x, projected_y, projected_z = beam_intersection((x_val, y_val, z_val), pitch_angle_rad[i], roll_offset, angle_offset, search_step)

                # Clip the projected coordinates to avoid out-of-bounds indexing
                projected_x_clipped = np.clip(int(projected_x), 0, terrain_x.shape[1] - 1)
                projected_y_clipped = np.clip(int(projected_y), 0, terrain_y.shape[0] - 1)

                # Get the terrain height at the projected point
                terrain_height = terrain_z[projected_y_clipped, projected_x_clipped]

                # Calculate the Euclidean distance between the trajectory point and the projected terrain point
                distance = np.sqrt((x_val - projected_x) ** 2 + (y_val - projected_y) ** 2 + (z_val - terrain_height) ** 2)

                # Update the minimum distance for the first beam
                if beam_idx == 0:
                    if distance < min_distance_1:
                        min_distance_1 = distance
                        beam_length_1[i] = distance

                # Update the minimum distance for the second beam
                if beam_idx == 1:
                    if distance < min_distance_2:
                        min_distance_2 = distance
                        beam_length_2[i] = distance

    return beam_length_1, beam_length_2



beam_lengths_1, beam_lengths_2 = calculate_beam_length(trajectory_x, trajectory_y, trajectory_z, x, y, z, pitch_angle_deg, row_angle_deg)

# Plot the results
plt.figure(figsize=(12, 6))

plt.plot(trajectory_x, beam_lengths_1, label='Beam 1', color='blue')
plt.plot(trajectory_x, beam_lengths_2, label='Beam 2', color='red')

plt.xlabel('X-coordinate')
plt.ylabel('Beam Length (m)')
plt.title('Beam Length vs. X-coordinate')
plt.legend()

plt.show()

def calculate_true_altitude(beam_lengths_1, beam_lengths_2, pitch_angle_deg, roll_angle_deg):
    # Convert angles to radians
    pitch_angle_rad = np.radians(pitch_angle_deg)
    roll_angle_rad = np.radians(roll_angle_deg)
    theta1_rad = np.radians(-15)
    theta2_rad = np.radians(15)
    
    # Calculate the true altitude using the given formula
    h_true = (beam_lengths_1 * np.cos(theta1_rad) + 
              beam_lengths_2 * np.cos(theta2_rad) - 
              beam_lengths_1 * np.cos(theta1_rad - roll_angle_rad + pitch_angle_rad) - 
              beam_lengths_2 * np.cos(theta2_rad + roll_angle_rad + pitch_angle_rad)) + 50

    return h_true

# Call the function and calculate the true altitude
true_altitude = calculate_true_altitude(beam_lengths_1, beam_lengths_2, pitch_angle_deg, row_angle_deg)

# Plot the results
plt.figure(figsize=(12, 6))

plt.plot(trajectory_x, true_altitude-20, label='RA Altitude', color='red')
#plt.plot(trajectory_x, trajectory_z, label='Actual Altitude', color='blue')
plt.legend(loc='best')
plt.xlabel('X-coordinate')
plt.ylabel('Altitude (m)')
plt.title('RA Altitude')
plt.show()

def plot_trajectory_and_beams(terrain_x, terrain_y, terrain_z, trajectory_x, trajectory_y, trajectory_z, beam_lengths_1, beam_lengths_2, pitch_angle_deg, roll_angle_deg, step=100):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the terrain surface
    ax.plot_surface(terrain_x, terrain_y, terrain_z, cmap='terrain', alpha=0.4)

    # Plot the UAV's trajectory
    ax.plot(trajectory_x, trajectory_y, trajectory_z, color='blue', linewidth=2)

    # Convert angles to radians
    pitch_angle_rad = np.radians(pitch_angle_deg)
    roll_angle_rad = np.radians(roll_angle_deg)
    theta1_rad = np.radians(-15)
    theta2_rad = np.radians(15)

    # Plot the beams at specific points along the trajectory
    for i in range(0, len(trajectory_x), step):
        x_val = trajectory_x[i]
        y_val = trajectory_y[i]
        z_val = trajectory_z[i]

        # Calculate the end points of the beams
        end_x_1 = x_val + beam_lengths_1[i] * np.sin(theta1_rad - roll_angle_rad[i] + pitch_angle_rad[i])
        end_y_1 = y_val + beam_lengths_1[i] * np.cos(theta1_rad - roll_angle_rad[i] + pitch_angle_rad[i])
        end_z_1 = z_val - beam_lengths_1[i] * np.cos(roll_angle_rad[i])

        end_x_2 = x_val + beam_lengths_2[i] * np.sin(theta2_rad + roll_angle_rad[i] + pitch_angle_rad[i])
        end_y_2 = y_val + beam_lengths_2[i] * np.cos(theta2_rad + roll_angle_rad[i] + pitch_angle_rad[i])
        end_z_2 = z_val - beam_lengths_2[i] * np.cos(roll_angle_rad[i])

        # Plot the beams
        ax.plot([x_val, end_x_1], [y_val, end_y_1], [z_val, end_z_1], color='red', linestyle='dashed', alpha=0.6)
        ax.plot([x_val, end_x_2], [y_val, end_y_2], [z_val, end_z_2], color='red', linestyle='dashed', alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('UAV Trajectory, Terrain, and Altimeter Beams')

    plt.show()

# Call the function to plot the trajectory and beams
plot_trajectory_and_beams(x, y, z, trajectory_x, trajectory_y, trajectory_z, beam_lengths_1, beam_lengths_2, pitch_angle_deg, row_angle_deg)
