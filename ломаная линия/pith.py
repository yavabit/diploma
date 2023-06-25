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

flight_trajectory = y + 30
flight_trajectory_Y = np.full(len(x), 500)  # середина по оси Y

height_above_terrain = flight_trajectory - Z

# Визуализация
""" fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='terrain')
ax.plot(x, flight_trajectory_Y, flight_trajectory, color='red', linewidth=2, label='Траектория полета')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Модель рельефа местности")
ax.set_box_aspect([3, 1, 1])
plt.show() """

# Учет тангажа БПЛА
pitch_angle = np.zeros(2000)
h_altimeter = np.zeros(2000)

for i in range(2000):
    if(x[i] < 250):
        pitch_angle[i] = terrain_angle_1
        h_altimeter[i] = np.cos(np.radians(terrain_angle_1)) * 30
    if(x[i] > 250 and x[i] < 500):
        pitch_angle[i] = terrain_angle_2
        h_altimeter[i] = np.cos(np.radians(terrain_angle_2)) * 30
    if(x[i] > 500 and x[i] <= 1750):
        pitch_angle[i] = terrain_angle_3
        h_altimeter[i] = np.cos(np.radians(terrain_angle_3)) * 30
    if(x[i] > 1750 and x[i] <= 2000):
        pitch_angle[i] = terrain_angle_4
        h_altimeter[i] = np.cos(np.radians(terrain_angle_4)) * 30
    if(x[i] == 500 or x[i] == 250 ):
        pitch_angle[i] =0
        h_altimeter[i] = 30
        
pitch_angle = gaussian_filter(pitch_angle, sigma=18) 

for i in range(2000):
    if(x[i] < 250):
        h_altimeter[i] = np.cos(np.radians(pitch_angle[i])) * 30
    if(x[i] > 250 and x[i] < 500):
        h_altimeter[i] = np.cos(np.radians(pitch_angle[i])) * 30
    if(x[i] > 500 and x[i] <= 1750):
        h_altimeter[i] = np.cos(np.radians(pitch_angle[i])) * 30
    if(x[i] > 1750 and x[i] <= 2000):
        h_altimeter[i] = np.cos(np.radians(pitch_angle[i])) * 30
    if(x[i] == 500 or x[i] == 250 ):
        pitch_angle[i] =0
        h_altimeter[i] = 30
      
h_altimeter = gaussian_filter(h_altimeter, sigma=3)   
""" plt.plot(x, pitch_angle)
plt.xlabel("X")
plt.ylabel("Угол тангажа")
plt.title("Тангаж")
plt.grid(True)
plt.show() """

pitch_angle = np.radians(pitch_angle)
# Визуализация значений радиовысотомера
""" plt.plot(x, h_altimeter)
plt.xlabel("X")
plt.ylabel("Высота")
plt.title("Значения радиовысотомера с учетом тангажа")
plt.grid(True)
plt.show() """

# Задайте максимальное значение шума в метрах
max_noise = 0.3

noise_rv1 = np.random.uniform(-max_noise, max_noise, size=len(h_altimeter))

#h_rv1 = gaussian_filter(h_altimeter, sigma=3)
h_altimeter = h_altimeter + noise_rv1
h_altimeter = gaussian_filter(h_altimeter, sigma=3)
# Визуализация значений радиовысотомера
""" plt.plot(x, h_altimeter)
plt.xlabel("X")
plt.ylabel("Высота")
plt.title("Значения радиовысотомера с учетом шума")
plt.grid(True)
plt.show() """

plt.figure(figsize=(10, 6))
plt.plot(x, h_altimeter, label='Показания РВ')
plt.axhline(30, color='r', linestyle='--', label='Эталонная высота')
plt.xlabel('X (м)')
plt.ylabel('Y (м)')
plt.legend()
plt.grid(True)
# Добавляем текстовое поле с расчетом среднеквадратического отклонения
plt.show()

differ = 30 - h_altimeter
plt.plot(x, differ)
plt.xlabel("X (м)")
plt.ylabel("Y (м)")
plt.grid(True)
plt.show()

std_dev = np.std(h_altimeter - 30)




