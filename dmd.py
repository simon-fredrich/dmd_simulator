import numpy as np
from mirror import Mirror

class DMD:
    '''
    Class for creating a DMD consisting of an array of flat
    mirrors which are able to rotate along their diagonal 
    axis.
    '''
    def __init__(self, mirror_nr_x, mirror_nr_y, mirror_width, mirror_height, gap_x, gap_y) -> None:
        '''
        Constructor for DMD
        '''
        self.mirror_nr_x = mirror_nr_x # amount of mirrors in the x direction
        self.mirror_nr_y = mirror_nr_y # amount of mirrors in the y direction
        self.mirror_width = mirror_width # width of the mirrors
        self.mirror_height = mirror_height # height of the mirrors
        self.gap_x = gap_x
        self.gap_y = gap_y
        self.dmd_width = mirror_nr_x * (mirror_width + gap_x) - gap_x
        self.dmd_height = mirror_nr_y * (mirror_height + gap_y) - gap_y

    def range_check(self, mirror_x, mirror_y, tilt_angle, s, t):
        '''
        Check if parameters of mirror are within the defined range of the dmd.
        '''
        if (mirror_x < 0 or mirror_x >= self.mirror_nr_x):
            print("mirror_x has to be between 0 and {}".format(self.mirror_nr_x))
        if (mirror_y < 0 or mirror_y >= self.mirror_nr_y):
            print("mirror_y has to be between 0 and {}".format(self.mirror_nr_y))
        if (np.abs(tilt_angle) > 90):
            print("Tilt angle has to be inside [-90, 90]")
        if (s < 0 or s > self.mirror_width):
            print("s has to be inside [0, {}]".format(self.mirror_width))
        if (t < 0 or t > self.mirror_width):
            print("t has to be inside [0, {}]".format(self.mirror_height))

    def get_x(self, mirror_nr_x, mirror_nr_y, tilt_angle, s, t):
        '''
        Calculate the x coordinate depending on mirror properties
        '''
        self.range_check(mirror_nr_x, mirror_nr_y, tilt_angle, s, t)
        gamma = (2 * np.pi * tilt_angle) / 360
        cos = np.cos(gamma)
        sin = np.sin(gamma)
        # matrix_product = s * (0.5 * (1 - cos) + cos) + t * (0.5 * (1-cos) - 1/np.sqrt(2) * sin)
        matrix_product = s * (0.5 * (1 - cos) + cos) + t * (0.5 * (1-cos))
        return matrix_product + (self.mirror_width + self.gap_x) * mirror_nr_x - self.dmd_width / 2


nrY = 1080
nrX = 1920
latticeConstant = 7.56
fillFactor = 0.92
mirrorSize = np.sqrt(latticeConstant*latticeConstant*fillFactor)
gap = latticeConstant-mirrorSize
dmd = DMD(nrX, nrY, mirrorSize, mirrorSize, gap, gap)
print(dmd.get_x(0, 0, 40, 2, 2))
