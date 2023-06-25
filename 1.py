from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Открываем файл HGT C:\универ\ДИПЛОМ\Python\N50E037.SRTMGL1.hgt\N50E037.hgt
filename = 'N50E037.SRTMGL1.hgt/N50E037.hgt'
dataset = gdal.Open(filename, gdal.GA_ReadOnly)

# Получаем ширину и высоту файла HGT
width = dataset.RasterXSize
height = dataset.RasterYSize

# Получаем массив высотных данных
band = dataset.GetRasterBand(1)
elevation = band.ReadAsArray(0, 0, width, height)

# Применяем фильтр Гаусса для сглаживания модели рельефа
elevation_smoothed = gaussian_filter(elevation, sigma=3)

# Создаем график модели рельефа
plt.figure(figsize=(10,10))
plt.imshow(elevation_smoothed, cmap='terrain')
plt.colorbar()
plt.title('Relief Model of Belgorod Region')
plt.show()

# Выбираем диагональную линию в массиве высотных данных
diag_elevation = np.diagonal(elevation_smoothed)

# Создаем массив расстояний
distance = np.linspace(0, 86, len(diag_elevation))

# Создаем график зависимости высоты от расстояния
plt.figure(figsize=(10,5))
plt.plot(distance, diag_elevation)
plt.title('Height vs Distance')
plt.xlabel('Distance (km)')
plt.ylabel('Height (m)')
plt.show()

# Закрываем файл HGT
dataset = None
# Масштаб 86 на 86 км