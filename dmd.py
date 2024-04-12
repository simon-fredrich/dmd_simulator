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
        if (s < 0 or s > self.mirror_width):
            print("s has to be inside [0, {}]".format(self.mirror_width))
        if (t < 0 or t > self.mirror_width):
            print("t has to be inside [0, {}]".format(self.mirror_height))

    def get_x(self, mirror_nr_x, mirror_nr_y, tilt_angle, s, t):
        '''
        Calculate the x coordinate depending on mirror properties
        '''
        self.range_check(mirror_nr_x, mirror_nr_y, tilt_angle, s, t)
        gamma = (2 * np.pi * tilt_angle) / 360 # in radiants
        cos = np.cos(gamma)
        sin = np.sin(gamma)
        # matrix_product = s * (0.5 * (1 - cos) + cos) + t * (0.5 * (1-cos) - 1/np.sqrt(2) * sin)
        matrix_product = s * (0.5 * (1 - cos) + cos) + t * (0.5 * (1-cos))
        return matrix_product + (self.mirror_width + self.gap_x) * mirror_nr_x - self.dmd_width / 2
    
    def get_y(self, mirror_nr_x, mirror_nr_y, tilt_angle, s, t):
        '''
        Calculate the y coordinate depending on mirror properties
        '''
        self.range_check(mirror_nr_x, mirror_nr_y, tilt_angle, s, t)
        gamma = (2 * np.pi * tilt_angle) / 360 # in radiants
        cos = np.cos(gamma)
        sin = np.sin(gamma)
        # matrix_product = s * (0.5 * (1 - cos) + 1/np.sqrt(2) * sin) + t * (0.5 * (1 - cos) + cos)
        matrix_product = s * (0.5 * (1 - cos)) + t * (0.5 * (1-cos) + cos)
        return matrix_product + (self.mirror_width + self.gap_y) * mirror_nr_y - self.dmd_height / 2
    
    def get_z(self, mirror_nr_x, mirror_nr_y, tilt_angle, s, t):
        '''
        Calculate the z coordinate depending on mirror properties
        '''
        self.range_check(mirror_nr_x, mirror_nr_y, tilt_angle, s, t)
        gamma = (2 * np.pi * tilt_angle) / 360 # in radiants
        cos = np.cos(gamma)
        sin = np.sin(gamma)
        # matrix_product = s * (0.5 * (1 - cos) - 1/np.sqrt(2) * sin) + t * (0.5 * (1 - cos) + 1/np.sqrt(2) * sin)
        matrix_product = 1/np.sqrt(2) * (- s * sin + t * sin)
        return matrix_product
    
    def get_coordinates(self, mirror_nr_x, mirror_nr_y, tilt_angle, s, t):
        x = self.get_x(mirror_nr_x, mirror_nr_y, tilt_angle, s, t)
        y = self.get_y(mirror_nr_x, mirror_nr_y, tilt_angle, s, t)
        z = self.get_z(mirror_nr_x, mirror_nr_y, tilt_angle, s, t)
        return np.array([x, y, z])
    
    def show_surface(self, tilt_angle):
        '''
        Display surface of the dmd where all mirrors are in the 
        same state.
        '''
        params_per_mirror = 4
        width = int(self.mirror_nr_x * (self.mirror_width + self.gap_x))
        height = int(self.mirror_nr_y * (self.mirror_height + self.gap_y))
        print(width, height)
        print(self.dmd_width, self.dmd_height)
        print(self.get_coordinates(0, 0, 12, 3, 4))
        surface = np.zeros((width, height), np.double)
        x_values = np.array([])
        y_values = np.array([])
        z_values = np.array([])

        for m_x in np.arange(self.mirror_nr_x):
            for m_y in np.arange(self.mirror_nr_y):
                for s in np.arange(params_per_mirror):
                    s_i = self.mirror_width / params_per_mirror * s
                    for t in np.arange(params_per_mirror):
                        t_i = self.mirror_height / params_per_mirror * t
                        x = self.get_x(m_x, m_y, tilt_angle, s_i, t_i)
                        x_values = np.append(x_values, x)
                        y = self.get_y(m_x, m_y, tilt_angle, s_i, t_i)
                        y_values = np.append(y_values, y)
                        z = self.get_z(m_x, m_y, tilt_angle, s_i, t_i)
                        z_values = np.append(z_values, z)
                        # print(x, y, z)

        plt.scatter(x_values, y_values, c=z_values, cmap='viridis', s=20)
        plt.colorbar(label='Z values')
        plt.show()



def main():
    mirror_nr_x = 6
    mirror_nr_y = 6
    mirror_width = 5
    mirror_height = 5
    gap = 0.5
    dmd = DMD(mirror_nr_x, mirror_nr_y, mirror_width, mirror_height, gap, gap)
    dmd.show_surface(-45)

if __name__ == "__main__":
    main()
