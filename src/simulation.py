import numpy as np
from matplotlib import pyplot as plt

from dmd import Dmd1d, Dmd2d, save_surface

'''Below is the simulation for 1d mirrors.'''

class Simulation1d:
    def __init__(self, dmd:Dmd1d, incident_angle, wavelength, field_dimensions: tuple, res, source_type) -> None:
        self.dmd = dmd

        # wave
        self.k = 2 * np.pi / wavelength
        self.incident_angle_deg = incident_angle
        self.incident_angle_rad = np.deg2rad(incident_angle)
        self.k_vector = - self.k * np.array([np.cos(self.incident_angle_rad), np.sin(self.incident_angle_rad)])

        self.source_type = source_type
        self.res = res
        self.pixels_x = res * field_dimensions[0]
        self.pixels_y = res * field_dimensions[1]
        

        # define the range where the field should be calculated
        self.x_range = np.linspace(-field_dimensions[0]/2, field_dimensions[0]/2, self.pixels_x)
        self.y_range = np.linspace(0, field_dimensions[1], self.pixels_y) - self.dmd.mirror_size/2
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)

        self.E_incident = np.exp(1j * self.k * (self.X * np.cos(self.incident_angle_rad) + self.Y * np.sin(self.incident_angle_rad)))

    def get_phase_shift_mirrors(self, nr_mirror_x1, nr_mirror_x2):
        x1 = self.dmd.get_x(nr_mirror_x1, 0)
        y1 = self.dmd.get_x(nr_mirror_x1, 0)
        x2 = self.dmd.get_x(nr_mirror_x2, 0)
        y2 = self.dmd.get_x(nr_mirror_x2, 0)
        r1 = np.array([x1, y1])
        r2 = np.array([x2, y2])
        kx = np.array([0, self.k_vector[1]])  # projection of k_vector onto the x-axis
        return np.dot(kx, r2-r1)
    
    def get_phase_shift_point_sources(self, nr_mirror_x):
        x1 = self.dmd.get_x(nr_mirror_x, 0)
        x2 = self.dmd.get_x(nr_mirror_x, self.dmd.mirror_size)
        y1 = self.dmd.get_x(nr_mirror_x, 0)
        y2 = self.dmd.get_x(nr_mirror_x, self.dmd.mirror_size)
        r0 = np.array([x2-x1, y2-y1])  # vector inside mirror plane
        r0_norm = r0/np.linalg.norm(r0)
        km = np.dot(self.k_vector, r0_norm)*r0_norm # projection of k_vector onto the 0th mirror
        return np.dot(km, r0_norm)*np.linalg.norm(r0)/self.dmd.nr_sources_per_mirror
    
    def get_phases(self):
        delta_phase_m = self.get_phase_shift_mirrors(0, 1)

    def get_E_reflected(self, r, phase_shift):
        return np.exp(1j * (self.k * r + phase_shift))

    def get_E_total(self):
        epsilon = 1e-10
        E_total = np.zeros_like(self.X, dtype=complex)

        for nr_x in range(self.dmd.nr_x):
            k_proj = self.dmd.get_projection(nr_x, self.k, self.incident_angle_rad)
            for s in self.dmd.mirror_coords_x:
                r = np.sqrt(np.square(self.X - self.dmd.get_x(nr_x, s)) + np.square(self.Y - self.dmd.get_y(nr_x, s)))
                phase_shift = self.dmd.get_phase_shift(nr_x, s, k_proj)
                #phase_shift = self.dmd.get_phase_shift_old(nr_x, s, self.k, self.incident_angle_rad)

                # different fields based on the source type
                if self.source_type == "spherical":
                    E_total += self.get_E_reflected(r, phase_shift) / (r + epsilon)
                elif self.source_type == "plane":
                    E_total += self.get_E_reflected(r, phase_shift)
        
        return E_total
    

class Simulation2d:
    def __init__(self, dmd:Dmd2d, phi, theta, wavelength, field_dimensions: tuple, res, source_type) -> None:
        self.dmd = dmd

        # angles for defining wavevector in spherical coordinates
        self.phi_deg = phi
        self.phi_rad = np.deg2rad(phi)
        self.theta_deg = theta
        self.theta_rad = np.deg2rad(theta)
        self.k = 2 * np.pi / wavelength
        self.k_vector = self.k * \
        np.array([np.sin(self.theta_rad)*np.cos(self.phi_rad),
                  np.sin(self.theta_rad)*np.sin(self.phi_rad), 
                  np.cos(self.theta_rad)])
        self.source_type = source_type
        self.res = res  # resolution
        self.pixels_x = res * field_dimensions[0] 
        self.pixels_y = res * field_dimensions[1]

        # define the range where the field should be calculated
        self.x_range = np.linspace(-field_dimensions[0]/2, field_dimensions[0]/2, self.pixels_x)
        self.y_range = np.linspace(-field_dimensions[0]/2, field_dimensions[0]/2, self.pixels_y)
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)
        self.Z = self.dmd.mirror_size

        self.E_incident = np.exp(1j * self.k * (self.X*np.sin(self.theta_rad)*np.cos(self.phi_rad) + self.Y*np.sin(self.theta_rad)*np.sin(self.phi_rad) + self.Z*np.cos(self.theta_rad)))

    def get_E_reflected(self, r, phase_shift):
        return np.exp(1j * (self.k * r + phase_shift))

    def get_E_total(self):
        epsilon = 1e-10
        E_total = np.zeros_like(self.X, dtype=complex)

        for nr_x in range(self.dmd.nr_x):
            for nr_y in range(self.dmd.nr_y):
                r0 = self.dmd.get_r0(nr_x, nr_y)  # vector pointing to the main edge of the mirror
                k_proj = self.dmd.get_projection(nr_x, nr_y, self.k_vector, r0)  # projection of wavevector onto plane
                # print(k_proj)
                for s in self.dmd.mirror_coords_x:
                    for t in self.dmd.mirror_coords_y:
                        r = np.sqrt(np.square(self.X - self.dmd.get_x(nr_x, nr_y, s, t)) + 
                                    np.square(self.Y - self.dmd.get_y(nr_x, nr_y, s, t)) + 
                                    np.square(self.Y - self.dmd.get_z(nr_x, nr_y, s, t)))  # vector pointing to point source
                        # print(type(r), r)

                        phase_shift = self.dmd.get_phase_shift(nr_x, nr_y, s, t, k_proj, r0)  # phase shift of point source
                        # print(phase_shift)

                        # different fields based on the source type
                        if self.source_type == "spherical":
                            E_total += self.get_E_reflected(r, phase_shift) / (r + epsilon)
                        elif self.source_type == "plane":
                            E_total += self.get_E_reflected(r, phase_shift)
        
        return E_total


def show_intensities(intensities):
    fig, ax = plt.subplots(len(intensities), 1)
    for idx, intensity in enumerate(intensities):
        ax[idx].imshow(intensity, cmap='viridis', origin='lower')
        ax[idx].colorbar(label="z")
        ax[idx].xlabel("x")
        ax[idx].ylabel("y")


def main():
    pass


if __name__ == "__main__":
    main()
