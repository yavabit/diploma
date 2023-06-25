import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

# Terrain model
size = 100
scale = 50

x = np.linspace(0, 1000, size)
y = np.linspace(0, 1000, size)
x, y = np.meshgrid(x, y)

z = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        z[i, j] = -50 * np.exp(-((x[i, j] - 500)**2 + (y[i, j] - 500)**2) / 200000) + 50
# GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Terrain and UAV Simulation")
        self.geometry("800x600")

        self.pitch_value = tk.DoubleVar()
        self.roll_value = tk.DoubleVar()

        self.arrow_position = [0, 0]

        # 3D plot
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.plot_terrain()

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Sliders
        self.pitch_slider = tk.Scale(self, from_=-90, to=90, variable=self.pitch_value, command=self.update_plot, orient=tk.HORIZONTAL, label="Pitch")
        self.pitch_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.roll_slider = tk.Scale(self, from_=-90, to=90, variable=self.roll_value, command=self.update_plot, orient=tk.HORIZONTAL, label="Roll")
        self.roll_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.move_arrow()

    def plot_terrain(self):
        self.ax.clear()
        self.ax.set_title("Terrain and UAV")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.plot_surface(x, y, z, cmap="terrain", alpha=0.7)

    def update_plot(self, _):
        self.plot_terrain()
        self.plot_arrow()

    def move_arrow(self):
        # Update the arrow position
        self.arrow_position[0] += 1
        self.arrow_position[1] += 1

        if self.arrow_position[0] >= size:
            self.arrow_position[0] = 0

        if self.arrow_position[1] >= size:
            self.arrow_position[1] = 0

        self.plot_arrow()

        self.after(200, self.move_arrow)

    def plot_arrow(self):
        pitch = self.pitch_value.get()
        roll = self.roll_value.get()

        arrow_x = self.arrow_position[0]
        arrow_y = self.arrow_position[1]
        arrow_z = 50#z[arrow_x, arrow_y]

        arrow = Poly3DCollection(
            [
                [
                    (arrow_x, arrow_y, arrow_z),
                    (arrow_x - 10, arrow_y - 10, arrow_z),
                    (arrow_x - 10, arrow_y, arrow_z - 10),
                    (arrow_x, arrow_y - 10, arrow_z - 10),
                ]
            ],
            color="red",
        )
        self.ax.add_collection3d(arrow)
        self.canvas.draw()

app = App()
app.mainloop()
