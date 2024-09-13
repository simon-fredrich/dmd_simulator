import numpy as np
from numpy.fft import fftfreq, fftshift, fft2
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
        return self.mesh.real

    def abs(self) -> np.ndarray:
        return np.abs(self.mesh)
    
    def intensity(self) -> np.ndarray:
        return np.abs(self.mesh)**2
    
    def add(self, value) -> None:
        np.add(self.mesh, value, out=self.mesh)
    
    def multiply(self, value) -> None:
        np.multiply(self.mesh, value, out=self.mesh)

    def copy(self) -> 'ComplexField':
        copy=ComplexField(self.screen)
        copy.mesh=self.mesh
        return copy
    
    def shift(self, shift_x: float, shift_y: float) -> None:
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

    def shift_with_roll(self, shift_x: float, shift_y: float) -> None:
        """
        Shift the mesh by shift_x in the x direction and shift_y in the y direction.
        Positive values shift the mesh right/up, negative values shift the mesh left/down.
        """
        # Return early if no shift is needed
        if shift_x == 0 and shift_y == 0:
            return

        # Convert shift values to pixel values
        x_abs = np.abs(self.screen.x_max - self.screen.x_min)
        y_abs = np.abs(self.screen.y_max - self.screen.y_min)
        pixel_shift_x = int(self.screen.pixels * shift_x / x_abs)
        pixel_shift_y = int(self.screen.pixels * shift_y / y_abs)

        # Use numpy's roll function to shift the mesh
        self.mesh = np.roll(self.mesh, shift=(pixel_shift_y, pixel_shift_x), axis=(0, 1))

        # Handle boundary conditions (fill rolled over areas with zeros)
        if pixel_shift_x > 0:
            self.mesh[:, :pixel_shift_x] = 0
        elif pixel_shift_x < 0:
            self.mesh[:, pixel_shift_x:] = 0

        if pixel_shift_y > 0:
            self.mesh[:pixel_shift_y, :] = 0
        elif pixel_shift_y < 0:
            self.mesh[pixel_shift_y:, :] = 0

    def display(self, plot_type="abs", cmap="viridis") -> None:
        """Speichert das gegebene Feld als Bilddatei."""
        if plot_type == 'real':
            field_to_plot = self.real()
            title = "Real Part"
        elif plot_type == 'imag':
            field_to_plot = self.imag()
            title = "Imaginary Part"
        elif plot_type == 'abs':
            field_to_plot = self.abs()
            title = "Magnitude"
        elif plot_type == 'fft':
            fft, kx, ky = self.fft()
            title = "FFT Amplitude"
        elif plot_type == 'inten':
            field_to_plot = self.intensity()
            title = "Intensity"
        else:
            raise ValueError("Invalid plot_type. Choose 'real', 'imag', 'abs' or 'fft'.")

        plt.figure(figsize=(10, 8))
        plt.title(f"{title} of Field")
        if plot_type!="fft":
            plt.imshow(field_to_plot,
                    extent=(self.screen.x_min, self.screen.x_max,
                            self.screen.y_min, self.screen.y_max),
                    cmap=cmap)
        elif plot_type=="fft":
            plt.imshow(fft.abs(),
                    extent=(np.min(kx), np.max(kx),
                            np.min(ky), np.max(ky)),
                    cmap=cmap)
        plt.colorbar(label=title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        

    def fft(self, coeff=10) -> tuple['ComplexField', np.ndarray, np.ndarray]:
        """
        Perform a 2D Fourier transform on the complex field and return a tuple of:
        - A new ComplexField object containing the Fourier-transformed field.
        - The frequency axes in the x and y directions.
        """
        # Perform the 2D Fourier transform
        fft_field = fftshift(fft2(self.intensity(), s=[coeff*self.shape[0], coeff*self.shape[1]]))

        # Calculate the frequency axes
        # dx = (self.screen.x_max - self.screen.x_min) / self.screen.pixels
        # dy = (self.screen.y_max - self.screen.y_min) / self.screen.pixels

        freq_x = fftfreq(self.shape[1])#, d=dx)
        freq_y = fftfreq(self.shape[0])#, d=dy)

        # Shift the frequencies to match the transformed data
        # freq_x = fftshift(freq_x)
        # freq_y = fftshift(freq_y)

        # Create a new ComplexField to hold the transformed field
        transformed_field = ComplexField(self.screen)
        transformed_field.mesh = fft_field

        return transformed_field, freq_x, freq_y
    

if __name__ == "__main__":
    from dmd import Dmd3d
    from simulation import Simulation3d
    from metadata import MetaData

    meta=MetaData(tilt_angle_deg=12,
              incident_angle_deg=-24,
              m_size=10,
              m_gap=1,
              nr_m=512,
              nr_s=100,
              wavelength=0.5,
              pixels=64)

    x_min=-np.sqrt(2)/2*meta.d_size
    x_max=np.sqrt(2)/2*meta.d_size
    y_min=-np.sqrt(2)/2*meta.d_size
    y_max=np.sqrt(2)/2*meta.d_size
    z=0

    dmd=Dmd3d(meta)
    sim=Simulation3d(dmd, meta)