from metadata import MetaData
import numpy as np
import matplotlib.pyplot as plt

class Beam():
    def __init__(self, meta:MetaData) -> None:
        self.nr_m=meta.nr_m
    
    def amplitude(self, radius, type="flattop"):
        x, y = np.ogrid[0:self.nr_m:self.nr_m*1j, \
                        0:self.nr_m:self.nr_m*1j]
        
        if type.lower()=="flattop":
            r_squared=np.square(x-self.nr_m/2)+np.square(y-self.nr_m/2)
            flat_top=np.ones((self.nr_m, self.nr_m))
            flat_top[r_squared>np.square(radius)]=0
            return flat_top
        
    def display(self, values):
        plt.imshow(values)
        plt.colorbar()
        plt.show()