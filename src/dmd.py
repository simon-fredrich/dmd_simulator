import numpy as np
import matplotlib.pyplot as plt

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
        self.gap_x = gap_x # gap between mirrors in the x direction
        self.gap_y = gap_y # gap between mirrors in the y direction
        self.dmd_width = mirror_nr_x * (mirror_width + gap_x) - gap_x # total width of the dmd (x direction)
        self.dmd_height = mirror_nr_y * (mirror_height + gap_y) - gap_y # total height of the dmd (y direction)

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
        for s_i in s:
            if (s_i < 0 or s_i > self.mirror_width):
                print("s has to be inside [0, {}]".format(self.mirror_width))
        for t_i in t:
            if (t_i < 0 or t_i > self.mirror_width):
                print("t has to be inside [0, {}]".format(self.mirror_height))

    def get_x(self, mirror_nr_x, mirror_nr_y, tilt_angle, s, t):
        '''
        Calculate the x coordinates depending on mirror properties

        Returns: np.meshgrid
        '''
        self.range_check(mirror_nr_x, mirror_nr_y, tilt_angle, s, t)
        ss, tt = np.meshgrid(s, t)
        gamma = (2 * np.pi * tilt_angle) / 360 # in radiants
        cos = np.cos(gamma)
        sin = np.sin(gamma)
        matrix_product = ss * (0.5 * (1 - cos) + cos) + tt * (0.5 * (1-cos))
        return matrix_product + (self.mirror_width + self.gap_x) * mirror_nr_x - self.dmd_width / 2
    
    def get_y(self, mirror_nr_x, mirror_nr_y, tilt_angle, s, t):
        '''
        Calculate the y coordinates depending on mirror properties

        Returns: np.meshgrid
        '''
        self.range_check(mirror_nr_x, mirror_nr_y, tilt_angle, s, t)
        ss, tt = np.meshgrid(s, t)
        gamma = (2 * np.pi * tilt_angle) / 360 # in radiants
        cos = np.cos(gamma)
        sin = np.sin(gamma)
        matrix_product = ss * (0.5 * (1 - cos)) + tt * (0.5 * (1-cos) + cos)
        return matrix_product + (self.mirror_width + self.gap_y) * mirror_nr_y - self.dmd_height / 2
    
    def get_z(self, mirror_nr_x, mirror_nr_y, tilt_angle, s, t):
        '''
        Calculate the z coordinates depending on mirror properties

        Returns: np.meshgrid
        '''
        self.range_check(mirror_nr_x, mirror_nr_y, tilt_angle, s, t)
        ss, tt = np.meshgrid(s, t)
        gamma = (2 * np.pi * tilt_angle) / 360 # in radiants
        cos = np.cos(gamma)
        sin = np.sin(gamma)
        matrix_product = 1/np.sqrt(2) * (- ss * sin + tt * sin)
        return matrix_product
    
    def get_coordinates(self, mirror_nr_x, mirror_nr_y, tilt_angle, s, t):
        xx = self.get_x(mirror_nr_x, mirror_nr_y, tilt_angle, s, t)
        yy = self.get_y(mirror_nr_x, mirror_nr_y, tilt_angle, s, t)
        zz = self.get_z(mirror_nr_x, mirror_nr_y, tilt_angle, s, t)
        return np.array([xx, yy, zz])
    
    def show_surface(self, tilt_angle):
        '''
        Display surface of the dmd where all mirrors are in the 
        same state.
        '''

        params_per_mirror = 4 # s & t number per mirror (resolution)
        width = int(self.mirror_nr_x * (self.mirror_width + self.gap_x))
        height = int(self.mirror_nr_y * (self.mirror_height + self.gap_y))
        box = [-10, 10]

        # prerequisites for 3d-plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # iterate over mirrors
        for m_x in np.arange(self.mirror_nr_x):
            for m_y in np.arange(self.mirror_nr_y):
                s = np.linspace(0, params_per_mirror, box[1])
                t = np.linspace(0, params_per_mirror, box[1])
                
                # get coordinate-meshgrids for displaying mirror plane
                xx = self.get_x(m_x, m_y, tilt_angle, s, t)
                yy = self.get_y(m_x, m_y, tilt_angle, s, t)
                zz = self.get_z(m_x, m_y, tilt_angle, s, t)

                # add mirror plane to 3d-plot
                ax.plot_surface(xx, yy, zz, alpha=0.5)

        # show plot
        ax.set_zlim(box[0], box[1])
        plt.show()

def main():
    mirror_nr_x = 5
    mirror_nr_y = 5
    mirror_width = 5
    mirror_height = 5
    gap = 0.5
    dmd = DMD(mirror_nr_x, mirror_nr_y, mirror_width, mirror_height, gap, gap)
    dmd.show_surface(45)

if __name__ == "__main__":
    main()
