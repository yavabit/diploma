import os 
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

from tkinter import *
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox as mb

os.system('cls')

#Класс модели рельефа
class ReliefModel:
    def __init__(self, filename, diag_size):
        self.filename = filename
        self.distance = diag_size
    
    #Чтение данных файла
    def load_data(self):
        dataset = gdal.Open(self.filename, gdal.GA_ReadOnly)

        self.width = dataset.RasterXSize
        self.height = dataset.RasterYSize

        band = dataset.GetRasterBand(1)
        self.elevation = band.ReadAsArray(0, 0, self.width, self.height)
        
        dataset = None

    #Поулчение диагональных значений
    def get_diagonal(self, _sigma=100, plot_type='2d'):
        diagonal = np.diagonal(self.elevation)
        if(plot_type == '3d'):           
            diagonal = np.array([diagonal]*len(diagonal))
            diagonal = gaussian_filter(diagonal, sigma=_sigma)
        else:
            diagonal = np.diagonal(self.elevation)
            diagonal = gaussian_filter(diagonal, sigma=_sigma)
        return diagonal

    #График модели рельефа
    def plot_terrain(self):
        plt.figure(figsize=(10,10))
        plt.imshow(self.elevation, cmap='terrain')
        plt.colorbar()
        plt.title('Relief Model of Belgorod Region')
        plt.show()

    #3Д График модели рельефа
    def plot_3d(self, _sigma=100):
        diagonal = self.get_diagonal(_sigma, plot_type='3d')
        num_points = diagonal.shape[0]
        x = np.linspace(0, self.distance, num_points)
        y = np.linspace(0, self.distance, num_points)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X, Y, diagonal, cmap='terrain')
        ax.set_xlabel('Distance, km')
        ax.set_ylabel('Distance, km')
        ax.set_zlabel('Elevation, m')

        plt.show()

    #2Д График модели рельефа
    def plot_2d(self, _sigma=100):
        diagonal = self.get_diagonal(_sigma, plot_type='2d')
        num_points = diagonal.shape[0]
        distance = np.linspace(0, self.distance, num_points)

        plt.figure(figsize=(20,6))
        plt.xticks(np.arange(0, len(distance), 5))
        plt.plot(distance, diagonal)
        plt.grid()
        plt.title('Relief Model 2d (Smoothed: '+str(_sigma)+')')
        plt.xlabel('Distance (km)')
        plt.ylabel('Elevation (m)')

        plt.show()

    #Инициализация всех функций
    def run(self):
        self.load_data()

filename = 'N50E037.SRTMGL1.hgt/N50E037.hgt'
rm = ReliefModel(filename, 132)
rm.run() 

flight_height = 100  # Высота полета БПЛА над рельефом
angle_between_beams = 20  # Угол между лучами
distance_between_points = 30  # Расстояние между точками

def corrected_height(height_diff, pitch_angle, angle_between_beams):
    sin_pitch = np.sin(np.deg2rad(pitch_angle))
    sin_alpha = np.sin(np.deg2rad(angle_between_beams))
    correction_factor = 1 / (1 - sin_pitch * sin_alpha)
    corrected_diff = height_diff * correction_factor
    return corrected_diff

true_heights = rm.get_diagonal(100) + flight_height
terrain_heights = rm.get_diagonal(100)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="same")

def calculate_pitch_angles(true_heights, terrain_heights, lookahead=1, window_size=5, max_pitch_angle=20, route_length=132):
    height_diffs = true_heights[lookahead:] - true_heights[:-lookahead]
    num_points = len(true_heights)
    horizontal_distance = route_length / (num_points - 1)
    horizontal_diffs = np.full(height_diffs.shape, horizontal_distance * lookahead)
    pitch_angles = np.degrees(np.arctan(height_diffs / (horizontal_diffs + 1e-9))) 
    
    # Ограничиваем максимальный угол тангажа
    pitch_angles = np.clip(pitch_angles, -max_pitch_angle, max_pitch_angle)
    
    # Применяем скользящее среднее к углам тангажа
    smoothed_pitch_angles = moving_average(pitch_angles, window_size)
    
    # Добавляем начальные углы тангажа, равные нулю, чтобы суммарное количество углов было равно количеству точек
    padded_pitch_angles = np.hstack((smoothed_pitch_angles[:lookahead], smoothed_pitch_angles))
    return padded_pitch_angles

lookahead = 5
max_pitch_angle = 20
window_size = 105
pitch_angles = calculate_pitch_angles(true_heights, terrain_heights, lookahead, window_size, max_pitch_angle)

corrected_height_diffs = [corrected_height(true_height - terrain_height, pitch_angle, angle_between_beams)
                     for true_height, terrain_height, pitch_angle in zip(true_heights, terrain_heights, pitch_angles)]

corrected_heights = np.array([terrain_height + corrected_diff for terrain_height, corrected_diff in zip(terrain_heights, corrected_height_diffs)])

num_points = corrected_heights.shape[0]
distance = np.linspace(0, 132, num_points)
plt.figure(figsize=(20,6))
plt.xticks(np.arange(0, len(distance), 5))
plt.plot(distance, corrected_heights)
plt.plot(distance, rm.get_diagonal(100))
plt.plot(distance, true_heights)
plt.plot(distance, pitch_angles)
plt.grid()
plt.title('Relief Model 2d (Smoothed: '+str(100)+')')
plt.xlabel('Distance (km)')
plt.ylabel('Elevation (m)')
plt.show()

""" class App:    
    def btn1_click():
        filename = 'N50E037.SRTMGL1.hgt/N50E037.hgt'
        window = ttk.Toplevel()
        rm = ReliefModel(filename, 132)
        rm.run() 
        window.title('Modeling')
        window.geometry('500x350')    
    
        window.resizable(width=False, height=False)
        b1 = ttk.Button(window, text='2Д Модель', bootstyle=LIGHT, command=lambda: rm.plot_2d(5))
        b1.pack(side=TOP, padx=5, pady=5)

        b2 = ttk.Button(window, text='3Д Модель', bootstyle=LIGHT, command=lambda: rm.plot_3d())
        b2.pack(side=TOP, padx=5, pady=5)

        b3 = ttk.Button(window, text='Модель рельефа', bootstyle=LIGHT, command=lambda: rm.plot_terrain())
        b3.pack(side=TOP, padx=5, pady=5)        


    window = ttk.Window(themename='superhero')
    window.title('Modeling')
    window.geometry('800x550')    
    
    window.resizable(width=False, height=False)

    s = ttk.Style()
    s.configure('my.TButton', font=('Helvetica', 22))

    b1 = ttk.Button(window, text='Модель рельефа местности', bootstyle=LIGHT, style='my.TButton', command=btn1_click)
    b1.pack(side=TOP, padx=5, pady=5)

    b2 = ttk.Button(window, text='btn2', bootstyle=LIGHT, style='my.TButton')
    b2.pack(side=TOP, padx=5, pady=5)

    b3 = ttk.Button(window, text='btn3', bootstyle=LIGHT, style='my.TButton')
    b3.pack(side=TOP, padx=5, pady=5)

    window.mainloop()
    
t = App() """