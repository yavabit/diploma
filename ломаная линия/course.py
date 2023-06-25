import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

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
X, Y = np.meshgrid(x, np.linspace(0, 2000, 2000))
Z = np.tile(y, (2000, 1))

# Создание траектории полета
flight_trajectory_X = np.full(len(y), 1000)  # середина по оси X
flight_trajectory_Y = np.linspace(0, 2000, len(y))
flight_trajectory_Z = np.full(len(y), 74)

real_trajectory_X = np.full(len(y), 1000)  # середина по оси X
real_trajectory_Y = np.linspace(0, 2000, len(y))
real_trajectory_Z = np.full(len(y), 74)

j=0
for i in range(2000):
    if (i > 1000):
        j += 0.2
        real_trajectory_X[i] -= j

real_height_above_terrain = np.zeros(2000)

for i in range(2000):
    xi = real_trajectory_X[i]
    yi = real_trajectory_Y[i]
    zi = real_trajectory_Z[i]

    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
    terrain_Z = f(xi)

    real_height_above_terrain[i] = zi - terrain_Z

print(real_height_above_terrain)

""" height_above_terrain = flight_trajectory - Z """

# Визуализация
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.5)
ax.plot(flight_trajectory_X, flight_trajectory_Y, flight_trajectory_Z, color='red', linewidth=2, label='Ожидаема траектория полета')
ax.plot(real_trajectory_X, real_trajectory_Y, real_trajectory_Z, color='blue', linewidth=2, label='Реальная траектория полета')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Модель рельефа местности")
ax.set_box_aspect([3, 3, 1])
plt.show()

h_altimeter = np.zeros(2000)
h_altimeter_ideal = np.zeros(2000)

for i in range(2000):
    h_altimeter_ideal[i] = 30

# Визуализация значений радиовысотомера
""" plt.plot(x, h_altimeter)
plt.xlabel("X")
plt.ylabel("Высота")
plt.title("Значения радиовысотомера с учетом тангажа")
plt.grid(True)
plt.show() """

# Задайте максимальное значение шума в метрах
max_noise = 0.3

h_altimeter = real_height_above_terrain

noise_rv1 = np.random.uniform(-max_noise, max_noise, size=len(h_altimeter))

#h_rv1 = gaussian_filter(h_altimeter, sigma=3)
h_altimeter = h_altimeter + noise_rv1
""" h_altimeter = gaussian_filter(h_altimeter, sigma=3) """

h_altimeter_ideal = h_altimeter_ideal + noise_rv1
""" h_altimeter_ideal = gaussian_filter(h_altimeter_ideal, sigma=3) """
# Визуализация значений радиовысотомера
""" plt.plot(x, h_altimeter_ideal)
plt.xlabel("Y")
plt.ylabel("Высота")
plt.grid(True)
plt.show()

plt.plot(Y, h_altimeter, 'r')
plt.xlabel("X")
plt.ylabel("Высота")
plt.title("Погрешность")
plt.grid(True)
plt.show() """

diff = (h_altimeter-h_altimeter_ideal)
plt.plot(Y, diff, 'r')
plt.xlabel("X")
plt.ylabel("Высота")
plt.title("Погрешность")
plt.grid(True)
plt.show()