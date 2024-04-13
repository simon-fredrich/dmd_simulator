import numpy as np
import matplotlib.pyplot as plt
import discretisedfield as df
from mpl_toolkits.mplot3d import Axes3D

class Plane_Wave:
    def __init__(self, E_0, k_0, theta_deg, phi_deg) -> None:
        self.E_0 = E_0
        self.theta_rad = 2*np.pi*theta_deg/360
        self.phi_rad = 2*np.pi*phi_deg/360
        self.wavelenght = 2*np.pi/k_0
        self.k = k_0 * np.array([np.sin(self.theta_rad)*np.cos(self.phi_rad),
                                 np.sin(self.theta_rad)*np.sin(self.phi_rad), 
                                 np.cos(self.theta_rad)])
    
    def get_field_at_pos(self, x, y, z):
        '''
        Calculates the field at a specified position (x, y, z) at t=0
        '''
        return self.E_0 * np.exp(1j*(self.k[0]*x
                                     +self.k[1]*y
                                     +self.k[2]*z))
    
    def display_field_xy(self):
        '''
        Display field in plane (x, y, 0)
        '''
        n = 20
        x = np.linspace(0, 10, n)
        y = np.linspace(0, 10, n)
        xs, ys = np.meshgrid(x, y, sparse=True)
        zs = np.zeros_like(xs)
        E = self.get_field_at_pos(xs, ys, zs)
        h = plt.contourf(x, y, E)
        plt.axis('scaled')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def display_field_xz(self):
        '''
        Display field in plane (x, 0, z)
        '''
        n = 20
        x = np.linspace(0, 10, n)
        z = np.linspace(0, 10, n)
        xs, zs = np.meshgrid(x, z, sparse=True)
        ys = np.zeros_like(xs)
        E = self.get_field_at_pos(xs, ys, zs)
        h = plt.contourf(x, z, E)
        plt.axis('scaled')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("z")
        plt.show()

    def display_field_yz(self):
        '''
        Display field in plane (0, y, z)
        '''
        n = 20
        y = np.linspace(0, 10, n)
        z = np.linspace(0, 10, n)
        ys, zs = np.meshgrid(y, z, sparse=True)
        xs = np.zeros_like(ys)
        E = self.get_field_at_pos(xs, ys, zs)
        h = plt.contourf(y, z, E)
        plt.axis('scaled')
        plt.colorbar()
        plt.xlabel("y")
        plt.ylabel("z")
        plt.show()
    

def main():
    my_plane = Plane_Wave(1, 1, 30, 10)
    my_plane.display_field_yz()


if __name__ == "__main__":
    main()