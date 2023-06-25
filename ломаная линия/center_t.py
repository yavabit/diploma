import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

def generate_terrain(x_start, x_end, angle, starting_height):
    x = np.linspace(x_start, x_end, 500)
    y = starting_height + np.tan(angle) * (x - x_start)
    return x, y

# Участок 1: Подъем
terrain_angle_1 = 10
x1, y1 = generate_terrain(0, 250, np.radians(terrain_angle_1), 0)
# Участок 2: Спуск
terrain_angle_2 = -10
x2, y2 = generate_terrain(250, 500, np.radians(terrain_angle_2), y1[-1])
# Участок 3: Подъем
terrain_angle_3 = 5
x3, y3 = generate_terrain(500, 1750, np.radians(terrain_angle_3), y2[-1])
# Участок 4: Подъем с большим углом
terrain_angle_4 = 10
x4, y4 = generate_terrain(1750, 2000, np.radians(terrain_angle_4), y3[-1])
# Объединение участков
x = np.concatenate([x1, x2, x3, x4])
y = np.concatenate([y1, y2, y3, y4])
# Создание сетки для 3D-графика
X, Y = np.meshgrid(x, np.linspace(0, 1000, 1000))
Z = np.tile(y, (1000, 1))

flight_trajectory_Z = 150
flight_trajectory_Y = np.full(len(x), 500)  # середина по оси Y

height_above_terrain = flight_trajectory_Z - Z[500, :] 

# Визуализация
""" fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='terrain')
ax.plot(x, flight_trajectory_Y, flight_trajectory_Z, color='red', linewidth=2, label='Траектория полета')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Модель рельефа местности")
ax.set_box_aspect([3, 1, 1])
plt.show() """


""" plt.plot(x, height_above_terrain)
plt.xlabel("X")
plt.ylabel("Высота")
plt.title("Значения радиовысотомера с учетом тангажа")
plt.grid(True)
plt.show() """

deltaH = np.zeros(len(height_above_terrain))
geomH = height_above_terrain
dna = 15
""" terrain_angle_2 = abs(terrain_angle_2) """
for i in range(len(height_above_terrain)):
    #if(i < 250):
    if(i < 400):
        deltaH[i] = geomH[i] * (1 - np.cos(np.radians(terrain_angle_1))/2 * (1/(np.cos(np.radians(terrain_angle_1 - dna/2)))+1/(np.cos(np.radians(terrain_angle_1 + dna/2)))))
    if(i > 400 and i < 600):
        deltaH[i] = geomH[i] * (1 - np.cos(np.radians(terrain_angle_2)) / np.cos(np.radians(terrain_angle_2 + dna/2)))
         #deltaH[i] = geomH[i] * (1 - np.cos(np.radians(terrain_angle_2))/2 * (1/(np.cos(np.radians(terrain_angle_2 - dna/2)))+1/(np.cos(np.radians(terrain_angle_2 + dna/2)))))
    if(i > 600 and i <= 1650):
        deltaH[i] = geomH[i] * (1 - np.cos(np.radians(terrain_angle_3))/2 * (1/(np.cos(np.radians(terrain_angle_3 - dna/2)))+1/(np.cos(np.radians(terrain_angle_3 + dna/2)))))
    if(i > 1650 and i < 1860):
        deltaH[i] = geomH[i] * (1 - np.cos(np.radians(terrain_angle_3)) / np.cos(np.radians(terrain_angle_3 + dna/2)))
    if(i > 1850 and i <= 2000):
        deltaH[i] = geomH[i] * (1 - np.cos(np.radians(terrain_angle_4))/2 * (1/(np.cos(np.radians(terrain_angle_4 - dna/2)))+1/(np.cos(np.radians(terrain_angle_4 + dna/2)))))
    if(i == 250 ):
        deltaH[i] = geomH[i] * (1 - np.cos(np.radians(0)) / np.cos(np.radians(0 - dna/2)))
 

deltaH = gaussian_filter(deltaH, sigma=25)  


max_noise = 0.3 
noise_rv1 = np.random.uniform(-max_noise, max_noise, size=len(deltaH))
deltaH = deltaH + noise_rv1
deltaH = gaussian_filter(deltaH, sigma=3)
        
plt.plot(deltaH)
plt.xlabel("X (м)")
plt.ylabel("Y (м)")
plt.grid(True)
plt.show()



