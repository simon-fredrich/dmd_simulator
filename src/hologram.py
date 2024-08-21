from metadata import MetaData
import numpy as np

class Hologram():
    def __init__(self, meta:MetaData) -> None:
        self.nr_m=meta.nr_m
        self.wavelength=meta.wavelength
        self.phase: float
        self.amplitude: float
        self.X, self.Y=np.ogrid[0:self.nr_m:self.nr_m*1j,\
                                0:self.nr_m:self.nr_m*1j]

    def create(self, phase, amplitude):
        self.phase=phase
        self.amplitude=amplitude
        H=1/2*1/2*np.sign(
            np.cos(2*np.pi/self.wavelength*(self.X+self.Y)+phase)-
            np.cos(np.arcsin(amplitude)))
        return H
        