import numpy as np
import matplotlib.pyplot as plt
from screen import Screen

class ComplexField:
    def __init__(self, screen: Screen) -> None:
        self.mesh=np.zeros_like(screen.X, dtype=complex)
        self.shape=self.mesh.shape
        self.screen=screen

    def imag(self):
        return self.mesh.imag
    
    def real(self):
        return self.mesh.imag

    def abs(self) -> np.ndarray:
        return np.abs(self.mesh)
    
    def intensity(self) -> np.ndarray:
        return np.abs(self.mesh)**2
    
    def add(self, value) -> None:
        self.mesh+=value
    
    def multiply(self, value) -> None:
        self.mesh*=value

    def copy(self) -> 'ComplexField':
        copy=ComplexField(self.screen)
        copy.mesh=self.mesh
        return copy
    
    def shift(self, shift_x: int, shift_y: int) -> None:
        """
        Shift the mesh by shift_x in the x direction and shift_y in the y direction.
        Positive values shift the mesh right/up, negative values shift the mesh left/down.
        """
        shifted_mesh = np.zeros_like(self.mesh)

        # Convert shift values to pixel values
        x_abs=np.abs(self.screen.x_max-self.screen.x_min)
        y_abs=np.abs(self.screen.y_max-self.screen.y_min)
        pixel_shift_x=int(self.screen.pixels*shift_x/x_abs)
        pixel_shift_y=int(self.screen.pixels*shift_y/y_abs)

        # Determine the slicing ranges based on the shift
        if pixel_shift_x > 0:  # Shift right
            start_x_src = 0
            end_x_src = self.shape[0] - pixel_shift_x
            start_x_dst = pixel_shift_x
            end_x_dst = self.shape[0]
        else:  # Shift left
            start_x_src = -pixel_shift_x
            end_x_src = self.shape[0]
            start_x_dst = 0
            end_x_dst = self.shape[0] + pixel_shift_x

        if pixel_shift_y > 0:  # Shift up
            start_y_src = pixel_shift_y
            end_y_src = self.shape[1]
            start_y_dst = 0
            end_y_dst = self.shape[1] - pixel_shift_y
        else:  # Shift down
            start_y_src = 0
            end_y_src = self.shape[1] + pixel_shift_y
            start_y_dst = -pixel_shift_y
            end_y_dst = self.shape[1]

        # Shift the mesh
        shifted_mesh[start_y_dst:end_y_dst, start_x_dst:end_x_dst] = \
            self.mesh[start_y_src:end_y_src, start_x_src:end_x_src]

        # Update the mesh with the shifted version
        self.mesh = shifted_mesh

        # Update the screen dimensions
        self.screen.x_max+=shift_x
        self.screen.x_min+=shift_x
        self.screen.y_max+=shift_y
        self.screen.y_min+=shift_y
        self.screen.update()

    def display(self) -> None:
        """ A simple method to visualize the real part of the field."""
        import matplotlib.pyplot as plt
        plt.imshow(self.real(), extent=(self.screen.x_min, self.screen.x_max, self.screen.y_min, self.screen.y_max))
        plt.colorbar()
        plt.title('Real part of ComplexField')
        plt.show()
    
    # def shift(self, shift_x, shift_y):
    #     self.screen.x_max+=shift_x
    #     self.screen.x_min+=shift_x
    #     self.screen.y_max+=shift_y
    #     self.screen.y_min+=shift_y
    #     self.screen.update()
