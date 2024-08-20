import numpy as np

class Screen:
    def __init__(self, pixels:int, x_min:float, x_max:float, y_min:float, y_max:float, z:float) -> None:
        self.pixels=pixels
        self.x_min, self.x_max=x_min, x_max
        self.y_min, self.y_max=y_min, y_max
        self.z=z
        self.x_range=np.linspace(x_min, x_max, pixels)
        self.y_range=np.linspace(y_min, y_max, pixels)
        self.X, self.Y=np.meshgrid(self.x_range, self.y_range)
        self.Z=np.ones_like(self.X)*z

    def update(self):
        self.x_range=np.linspace(self.x_min, self.x_max, self.pixels)
        self.y_range=np.linspace(self.y_min, self.y_max, self.pixels)
        self.X, self.Y=np.meshgrid(self.x_range, self.y_range)
        self.Z=np.ones_like(self.X)*self.z

    def properties(self) -> None:
        print("Screen Parameters: ")
        print(f"resolution: {self.pixels}x{self.pixels}")
        print(f"x: [{self.x_min}, {self.x_max}]")
        print(f"y: [{self.y_min}, {self.y_max}]")
        print(f"z: {self.z}")