import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

def generate_terrain(x_start, x_end, angle, starting_height):
    x = np.linspace(x_start, x_end, 500)
    print((x_end - x_start) / 499)
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

# Создание траектории полета
flight_trajectory_X = np.full(len(y), 1000)  # середина по оси X
flight_trajectory_Y = np.linspace(0, 1000, len(y))
terrain_height = 44
flight_trajectory_Z = np.full(len(y), terrain_height + 30)

# Визуализация
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.5)
ax.plot(flight_trajectory_X, flight_trajectory_Y, flight_trajectory_Z, color='red', linewidth=2, label='Траектория полета')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Модель рельефа местности")
ax.set_box_aspect([3, 3, 1])
plt.show()

# Учет крена БПЛА
roll_angle = np.zeros(1000)
h_altimeter = np.zeros(1000)

for i in range(1000):
    if(i >= 200 and i < 300):
        roll_angle[i] = -15
    if(i >= 400 and i < 500):
        roll_angle[i] = 15
        
    if(i >= 600 and i < 700):
        roll_angle[i] = -20
    if(i >= 800 and i < 900):
        roll_angle[i] = 20

roll_angle = gaussian_filter(roll_angle, sigma=10)

plt.plot(Y, roll_angle, 'r')
plt.xlabel("Y")
plt.ylabel("Крен")
plt.grid(True)
plt.show()

""" roll_angle = np.radians(abs(roll_angle)) """
terrain_angle_3 = np.radians(terrain_angle_3)

for i in range(1000):
    if(roll_angle[i] < 0):
        roll_angle[i] = np.radians((roll_angle[i]))
        angle_diff = np.pi / 2 + terrain_angle_3 - roll_angle[i]
        h_altimeter[i] = (30 * np.sin(np.pi / 2-terrain_angle_3)) / (np.sin(angle_diff))
    elif(roll_angle[i] >= 0):
        roll_angle[i] = np.radians((roll_angle[i]))
        angle_diff = np.pi / 2 - terrain_angle_3 - (roll_angle[i])
        h_altimeter[i] = (30 * np.sin(np.pi / 2+terrain_angle_3)) / (np.sin(angle_diff))

# Задайте максимальное значение шума в метрах
max_noise = 0.3

noise_rv1 = np.random.uniform(-max_noise, max_noise, size=len(h_altimeter))
h_altimeter = h_altimeter + noise_rv1
h_altimeter = gaussian_filter(h_altimeter, sigma=3)

plt.plot(Y, h_altimeter, 'r')
plt.xlabel("X (м)")
plt.ylabel("Y (м)")
plt.grid(True)
plt.show()

differ = h_altimeter - 30
plt.plot(Y, differ, 'r')
plt.xlabel("X (м)")
plt.ylabel("Y (м)")
plt.grid(True)
plt.show()

std_dev = np.std(h_altimeter - 30)

plt.figure(figsize=(10, 6))
plt.plot(Y, h_altimeter, 'r')
plt.axhline(30, color='r', linestyle='--', label='Эталонная высота')
plt.xlabel('X (м)')
plt.ylabel('Y (м)')
plt.legend()
plt.grid(True)
# Добавляем текстовое поле с расчетом среднеквадратического отклонения
plt.text(0.05, 0.05, f'Среднеквадратическое отклонение: {std_dev:.3f}', transform=plt.gca().transAxes, bbox=dict(facecolor='red', alpha=0.5))

plt.show()



