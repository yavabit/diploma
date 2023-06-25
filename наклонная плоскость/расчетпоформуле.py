import numpy as np

h_rv1 = float(input('длинна rv1: '))
h_rv2 = float(input('длинна rv2: '))
pitch = float(input('тангаж: '))

h_true = (h_rv1 * np.cos(np.radians(pitch)) + h_rv2 * np.cos(np.radians(pitch))) / 2
print(h_true)
