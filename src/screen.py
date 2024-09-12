import numpy as np

class Screen:
    def __init__(self, pixels:int, 
                 x_min:float, x_max:float,
                 y_min:float=None, y_max:float=None,
                 z_min:float=None, z_max:float=None, 
                 z_value:float=None, y_value:float=None) -> None:
        self.pixels=pixels
        self.x_min, self.x_max=x_min, x_max
        self.y_min, self.y_max=y_min, y_max
        self.z=z_value
        self.x_range=np.linspace(x_min, x_max, pixels)
        if all(i is None for i in (z_min, z_max, y_value)):
            self.y_range=np.linspace(y_min, y_max, pixels)
            self.X, self.Y=np.meshgrid(self.x_range, self.y_range)
            self.Z=np.ones_like(self.X)*z_value
            self.x_label="x"
            self.y_label="y"
            self.extent=(x_min, x_max, y_min, y_max)
        elif all(i is None for i in (y_min, y_max, z_value)):
            self.z_range=np.linspace(z_min, z_max, pixels)
            self.X, self.Z=np.meshgrid(self.x_range, self.z_range)
            self.Y=np.ones_like(self.X)*y_value
            self.x_label="x"
            self.y_label="z"
            self.extent=(x_min, x_max, z_min, z_max)
        else:
            ValueError("Screen must be either vertical or horizontal.")

        

    def properties(self) -> None:
        print("Screen Parameters: ")
        print(f"resolution: {self.pixels}x{self.pixels}")
        print(f"x: [{self.x_min}, {self.x_max}]")
        print(f"y: [{self.y_min}, {self.y_max}]")
        print(f"z: {self.z}")

class VerticalScreen:
    def __init__(self, pixels:int, x_min:float, x_max:float, z_min:float, z_max:float, y:float) -> None:
        self.pixels=pixels
        self.x_min, self.x_max=x_min, x_max
        self.z_min, self.z_max=z_min, z_max
        self.y=y
        self.x_range=np.linspace(x_min, x_max, pixels)
        self.z_range=np.linspace(z_min, z_max, pixels)
        self.X, self.Z=np.meshgrid(self.x_range, self.z_range)
        self.Y=np.ones_like(self.X)*y
        self.x_label="x"
        self.y_label="z"

    def update(self):
        self.x_range=np.linspace(self.x_min, self.x_max, self.pixels)
        self.z_range=np.linspace(self.z_min, self.z_max, self.pixels)
        self.X, self.Z=np.meshgrid(self.x_range, self.z_range)
        self.Y=np.ones_like(self.X)*self.y

    def properties(self) -> None:
        print("Screen Parameters: ")
        print(f"resolution: {self.pixels}x{self.pixels}")
        print(f"x: [{self.x_min}, {self.x_max}]")
        print(f"z: [{self.z_min}, {self.z_max}]")
        print(f"y: {self.y}")