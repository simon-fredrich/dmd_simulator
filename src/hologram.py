from metadata import MetaData
import numpy as np
import matplotlib.pyplot as plt

class Hologram():
    def __init__(self, meta:MetaData) -> None:
        self.nr_m=meta.nr_m
        self.wavelength=meta.wavelength
        self.phase: float
        self.amplitude: float
        self.X, self.Y=np.ogrid[0:self.nr_m:self.nr_m*1j,\
                                0:self.nr_m:self.nr_m*1j]
        self.H: np.ndarray

    def create(self, lambda_x, lambda_y, phase, amplitude):
        """
        Create a binary hologram with the given parameters.

        Parameters
        ----------
        lambda_x, lambda_y : float
            Spatial frequencies in the x and y directions.
        phase : float
            Phase of the hologram.
        amplitude : float
            Amplitude of the hologram.

        Returns
        -------
        H : numpy array
            The hologram pattern.
        """
        self.phase=phase
        self.amplitude=amplitude
        self.lambda_x=lambda_x
        self.lambda_y=lambda_y
        H=1/2+1/2*np.sign(
            np.cos(2*np.pi*(self.X/lambda_x+self.Y/lambda_y)+phase)-
            np.cos(np.arcsin(amplitude)))
        
        self.H=H
        return H
    
    def display(self):
        plt.imshow(self.H, cmap=plt.cm.gray)
        plt.show()
        

    def pattern_on(self):
        return np.ones((self.nr_m, self.nr_m))
    
    def pattern_off(self):
        return np.zeros((self.nr_m, self.nr_m))
    
    def grating(self, period):
        return 1/2+1/2*np.sign(np.cos(2*np.pi/period*(self.X+self.Y)))
    