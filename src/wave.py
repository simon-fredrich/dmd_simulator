import numpy as np
import matplotlib.pyplot as plt
import discretisedfield as df
from mpl_toolkits.mplot3d import Axes3D

class Plane:
    def __init__(self, E_0, k_0, wavelenght, incident_angle_degree) -> None:
        self.E_0 = E_0
        self.wavelenght = wavelenght
        self.incident_angle_deg = incident_angle_degree
        self.incident_angle_rad = 2 * np.pi * incident_angle_degree / 360
        self.k = k_0 * np.array([np.cos(self.incident_angle_rad), np.sin(self.incident_angle_rad)])
    
    def get_field_at_xy(self, position):
        '''
        Calculates the field at a specified point (x, y)
        '''
        return self.E_0 * np.sin(self.k[0]*position[0] + self.k[1]*position[1])
    
    def show_field_2D(self):
        lower_bound = -10
        upper_bound = 10
        p1 = (-10, -10, -1)
        p2 = (10, 10, 1)
        cell = (0.5, 0.5, 0.5)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        field = df.Field(mesh, nvdim=1, value=self.get_field_at_xy, norm=10)

        try:
            field.orientation.sel(z=0.5).mpl(figsize=(15, 12))
        except RuntimeError as e:
            print("Exception raised:", e)
        plt.show()

    def show_field_3D(self):
        # create the figure
        fig = plt.figure()

        # add axes
        ax = fig.add_subplot(111,projection='3d')

        xx, yy = np.meshgrid(np.linspace(0, 20, 20), np.linspace(0, 20, 20))
        z = self.get_field_at_xy((xx, yy))

        # plot the plane
        ax.plot_surface(xx, yy, z, alpha=0.5)

        plt.show()
    

def main():
    my_plane = Plane(1, 1, 200e-9, 55)

    my_plane.show_field_3D()


if __name__ == "__main__":
    main()