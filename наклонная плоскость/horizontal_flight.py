import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Параметры рельефа
start_height = 200 # Начальная высота
distance = 1000 # Дальность полета
length = 1000 # Размерность массивов
height = 200 # Перепад высот
resolution = 1 # Разрешение сетки, плостность точек, шаг в массиве координат

# Генерация массивов
x = np.arange(0, distance, resolution)
y = np.arange(0, length, resolution)
X, Y = np.meshgrid(x, y)

# Создание рельефа местности
a = height / distance
b = 0
c = start_height
Z = -a * X + b * Y + c

# Параметры траектории полета
trajectory_height = 220
trajectory_X = x
trajectory_Y = np.full(x.shape, length / 2)  # mid y-axis
trajectory_Z = np.full(x.shape, trajectory_height) # trajectory_Z = (-a * trajectory_X + b * trajectory_Y + c) - 20

# Истинная высота в каждой точке траектории
height_above_terrain = trajectory_Z - (-a * trajectory_X + b * trajectory_Y + c)
print(height_above_terrain)
# График рельефа и траектории
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='terrain', edgecolor='none')
ax.plot(trajectory_X, trajectory_Y, trajectory_Z, 'r', label='Flight trajectory')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_box_aspect([2, 1, 1])
ax.legend()
plt.show()

# Параметры высотомера
angle_between_rays = 30  # угол между лучами
angle_to_ground = angle_between_rays / 2 # фактический угол между вертикалью и лучом
angle_to_ground_rad = np.radians(angle_to_ground)

# Тангаж
pitch_angle = 0
pitch_angle = np.radians(pitch_angle)

# Рассчет длинны каждого луча высотомера
h_rv1 = height_above_terrain / np.cos(angle_to_ground_rad)  # крен отсутствует
h_rv2 = height_above_terrain / np.cos(angle_to_ground_rad)  # крен отсутствует

# Рассчет измеренной радиовысотомером высоты
h_altimeter = (h_rv1 * np.cos(pitch_angle) + h_rv2 * np.cos(pitch_angle)) / 2

# График истинной высоты и измеренной
plt.figure()
plt.plot(trajectory_X, height_above_terrain, 'r', label='Истинная высота над рельефом местности')
plt.plot(trajectory_X, h_altimeter, 'b', label='Измерения радиовысотомера')
plt.xlabel('Дистанция по оси Х')
plt.ylabel('Высота')
plt.legend()
plt.grid()
plt.show()

h_difference = abs(h_altimeter-height_above_terrain)

# График истинной высоты и измеренной
plt.figure()
plt.plot(trajectory_X, h_difference, 'r', label='Погрешность измерения')
plt.xlabel('Дистанция по оси Х')
plt.ylabel('Высота')
plt.legend()
plt.grid()
plt.show()


