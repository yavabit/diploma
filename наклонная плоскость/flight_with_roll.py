import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import os 

os.system('cls')
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
trajectory_X = x
# Создание траектории полета
trajectory_Y = np.zeros_like(trajectory_X)
for i, x_val in enumerate(trajectory_X):
    if x_val <= 100:
        trajectory_Y[i] = 500
    elif x_val >= 950:
        trajectory_Y[i] = 635
    else:
        # Уравнение прямой, проходящей через точки (100, 500) и (950, 235) y = ((y2 - y1)/(x2 - x1)) * (x - x1) + y1
        y_val = -(x_val - 100) * (500 - 635) / (950 - 100) + 500
        trajectory_Y[i] = y_val
trajectory_Z = 230

# Истинная высота в каждой точке траектории
height_above_terrain = trajectory_Z - (-a * trajectory_X + b * trajectory_Y + c)
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

course_angle = np.zeros(distance)

# Крен
roll_angles = np.zeros(distance)

j=0
n=0
for i in range(len(roll_angles)):
    if(100 <= i <= 494):
        j += 0.05
        course_angle[i] += j
        roll_angles[i] = -20
    if(494 < i < 556):
        course_angle[i] = j
    if(556 <= i <= 950):
        n += 0.05
        roll_angles[i] = 20
        course_angle[i] = (j-n)
        
# График истинной высоты и измеренной
plt.figure(figsize=(16, 8))
plt.plot(trajectory_X, course_angle, 'r', label='курс')
plt.plot(trajectory_X, roll_angles, 'b', label='крен')
plt.xlabel('Дистанция по оси Х')
plt.ylabel('Высота')
plt.legend()
plt.show()
print(course_angle)
roll_angles = np.radians(roll_angles)

h_rv1 = np.zeros(distance)
h_rv2 = np.zeros(distance)

# Рассчет длинны каждого луча высотомера
for i in range(distance):
    if(course_angle[i] == 0 and roll_angles[i] == 0):
        h_rv1[i] = height_above_terrain[i] / np.cos(angle_to_ground_rad) 
        h_rv2[i] = height_above_terrain[i] / np.cos(angle_to_ground_rad)        
        
    elif(course_angle[i] == 0 and roll_angles[i] != 0):
        h_rv1[i] = height_above_terrain[i] / np.cos(angle_to_ground_rad - roll_angles[i])
        h_rv2[i] = height_above_terrain[i] / np.cos(angle_to_ground_rad + roll_angles[i])
    else:
        # задаем точки на луче
        p1 = np.array([trajectory_X[i], trajectory_Y[i], 230]) # первая точка на луче
        
        # вторая точка на луче
        x2 = trajectory_X[i] + ((height_above_terrain[i] / np.cos(angle_to_ground_rad - roll_angles[i]))**2 - height_above_terrain[i]**2)**0.5
        y2 = trajectory_Y[i] + ((height_above_terrain[i] / np.cos(angle_to_ground_rad - roll_angles[i]))**2 - height_above_terrain[i]**2)**0.5 * np.tan(np.radians(course_angle[i]))
        z2 =  230 - height_above_terrain[i]
        p2 = np.array([x2, y2, z2])
       
        # задаем нормаль к плоскости
        n = np.array([a, -b, 1]) # коэффициенты уравнения плоскости: ax + by + cz + d = 0
        # находим направляющий вектор луча
        v = p2 - p1

        # находим координаты точки на плоскости, через которую проходит перпендикуляр к лучу из его начальной точки
        t = - (np.dot(n, p1) + (-c)) / np.dot(n, v)
       
        intersection_point = p1 + t * v
        # находим вектор, соединяющий начальную точку луча и найденную точку на плоскости
        
        h_rv1[i] = np.linalg.norm(intersection_point - p1)

        # вторая точка на луче
        x2 = trajectory_X[i] - ((height_above_terrain[i] / np.cos(angle_to_ground_rad - roll_angles[i]))**2 - height_above_terrain[i]**2)**0.5
        y2 = trajectory_Y[i] - ((height_above_terrain[i] / np.cos(angle_to_ground_rad - roll_angles[i]))**2 - height_above_terrain[i]**2)**0.5 * np.tan(np.radians(course_angle[i]))
        z2 =  230 - height_above_terrain[i]
        p2 = np.array([x2, y2, z2])

        # задаем нормаль к плоскости
        n = np.array([a, -b, 1]) # коэффициенты уравнения плоскости: ax + by + cz + d = 0
        # находим направляющий вектор луча
        v = p2 - p1
        # находим координаты точки на плоскости, через которую проходит перпендикуляр к лучу из его начальной точки
        t = - (np.dot(n, p1) + (-c)) / np.dot(n, v)
        intersection_point = p1 + t * v

        h_rv2[i] = np.linalg.norm(intersection_point - p1)

h_rv1 = gaussian_filter(h_rv1, sigma=3)
h_rv2 = gaussian_filter(h_rv2, sigma=3)
""" print(h_rv1[0])  """
# График истинной высоты и измеренной
plt.figure(figsize=(16, 8))
plt.plot(trajectory_X, h_rv1, 'r', label='правый луч')
plt.plot(trajectory_X, h_rv2, 'b', label='левый луч')
plt.xlabel('Дистанция по оси Х')
plt.ylabel('Высота')
plt.legend()
plt.grid()
plt.show()

# Рассчет измеренной радиовысотомером высоты
h_altimeter = (h_rv1 * np.cos(angle_to_ground_rad - roll_angles) + h_rv2 * np.cos(angle_to_ground_rad + roll_angles)) / 2
h_altimeter = gaussian_filter(h_altimeter, sigma=3)

# График истинной высоты и измеренной
plt.figure(figsize=(16, 8))
#plt.plot(trajectory_X, height_above_terrain, 'r', label='Истинная высота над рельефом')
plt.plot(trajectory_X, h_altimeter, 'b', label='Измерения радиовысотомера')
plt.xlabel('Дистанция по оси Х')
plt.ylabel('Высота')
plt.legend()
plt.grid()
plt.show()


