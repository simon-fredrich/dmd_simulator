import numpy as np
from matplotlib import pyplot as plt

from dmd import Dmd2d, Dmd3d

'''Below is the simulation for 1d mirrors.'''

class Simulation2d:
    def __init__(self, dmd:Dmd2d, incident_angle, wavelength, field_dimensions: tuple, res, source_type="spherical", phase_shift=True) -> None:
        # parameters concerning the dmd
        self.dmd = dmd
        self.dmd_source_pos = dmd.get_source_positions()
        self.dmd_source_x0 = [self.dmd_source_pos[0][i][0] for i in range(dmd.nr_sources_per_mirror)]  # source x-pos of 0th mirror
        self.dmd_source_y0 = [self.dmd_source_pos[0][i][1] for i in range(dmd.nr_sources_per_mirror)]  # source y-pos of 0th mirror
        self.rm = np.array([abs(self.dmd_source_x0[0]-self.dmd_source_x0[-1]),
                           abs(self.dmd_source_y0[0]-self.dmd_source_y0[-1])])  # vector inside mirror plane
        self.distance_between_sources = np.linalg.norm(self.rm)/dmd.nr_sources_per_mirror
        self.rm_norm = self.rm/np.linalg.norm(self.rm)  # normalized vector inside mirror plane+


        # wave
        self.k = 2 * np.pi / wavelength
        self.incident_angle_deg = incident_angle
        self.incident_angle_rad = np.deg2rad(incident_angle)
        self.k_vector = - self.k * np.array([np.cos(self.incident_angle_rad), np.sin(self.incident_angle_rad)])
        self.source_type = source_type
        self.phase_shift = phase_shift


        # calculate projections
        self.km = np.dot(self.k_vector, self.rm_norm) * self.rm_norm  # projection along mirror plane
        self.kx = np.dot(self.k_vector, np.array([1, 0])) * np.array([1, 0])  # projection along x-axis
        self.phase_shifts = self.get_total_phases()  # get all phase shifts in one big array


        # field parameters
        self.field_dimensions = field_dimensions
        self.res = res  # number of pixels for a single field dimension point
        self.pixels_x = res * field_dimensions[0]  # total number of pixels in x-direction
        self.pixels_y = res * field_dimensions[1]  # total number of pixels in y-direction
        

        # define the range where the field should be calculated
        self.x_range = np.linspace(-field_dimensions[0]/2, field_dimensions[0]/2, self.pixels_x)
        self.y_range = np.linspace(0, field_dimensions[1], self.pixels_y) - self.dmd.mirror_size/2
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)


        # calculate incoming field
        self.E_incident = np.exp(1j * self.k * (self.X * np.cos(self.incident_angle_rad) + self.Y * np.sin(self.incident_angle_rad)))


    def get_phase_shift_mirrors(self, nr_mirror_x1, nr_mirror_x2):
        # get position of 0th point-source for each mirror
        x1 = self.dmd_source_pos[nr_mirror_x1][0][0]
        y1 = self.dmd_source_pos[nr_mirror_x1][0][1]
        x2 = self.dmd_source_pos[nr_mirror_x2][0][0]
        y2 = self.dmd_source_pos[nr_mirror_x2][0][1]
        r12 = np.array([x2-x1, y2-y1])  # vector pointing from one mirror to the other
        return np.dot(self.kx, r12)  # return the phase shift of both mirrors
    
    def get_phase_shift_point_source(self, nr_mirror_x, nr_source):
        distance = self.distance_between_sources*nr_source
        mirror_phase = self.get_phase_shift_mirrors(0, nr_mirror_x)
        source_phase = mirror_phase + (np.dot(self.km, self.rm_norm) * distance)
        return source_phase  # return phase shift of 0th mirror and current point-source 
    
    def get_total_phases(self):
        # create array to hold phase shifts of point sources that are
        # categorized into the mirrors they are associated with
        total_phases = np.zeros((self.dmd.nr_mirrors_x, self.dmd.nr_sources_per_mirror))
        for nr_mirror_x in range(self.dmd.nr_mirrors_x):
            for nr_source in range(self.dmd.nr_sources_per_mirror):
                total_phases[nr_mirror_x, nr_source] = self.get_phase_shift_point_source(nr_mirror_x, nr_source)
        return total_phases
            

    def get_E_reflected(self, r, phase_shift):
        return np.exp(1j * (self.k * r + phase_shift))

    def get_E_total(self):
        epsilon = 1e-10

        # create array to hold field values
        E_total = np.zeros_like(self.X, dtype=complex)

        # loop over each point source and determine their phase shift
        # then add contribution to the total field based on source type
        for nr_mirror_x in range(self.dmd.nr_mirrors_x):
            for nr_source in range(self.dmd.nr_sources_per_mirror):
                xs = self.dmd_source_pos[nr_mirror_x][nr_source][0]
                ys = self.dmd_source_pos[nr_mirror_x][nr_source][1]
                r = np.sqrt(np.square(self.X - xs) + np.square(self.Y - ys))
                phase_shift = self.get_phase_shift_point_source(nr_mirror_x, nr_source)
                if not self.phase_shift:
                    phase_shift = 0

                # different fields based on the source type
                if self.source_type == "spherical":
                    E_total += self.get_E_reflected(r, phase_shift) / (r + epsilon)
                elif self.source_type == "plane":
                    E_total += self.get_E_reflected(r, phase_shift)
        
        return E_total

    def get_field_profile(self, field, y):
        # calculate the shift due to shifting the field
        # because I wanted to calculate the field a little below zero
        y_shift = y + self.dmd.mirror_size/2
        # mapping the real x-coordinate to the array index
        y_mapped = int(y_shift * self.pixels_y/self.field_dimensions[1])
        return field[y_mapped]  # return the field profile at set y-value
    
    def plot_field_profile(self, field, y):
        # plotting the field profile
        field_profile = self.get_field_profile(field, y)
        plt.plot(np.linspace(-self.field_dimensions[0]/2, self.field_dimensions[0]/2, len(field_profile)), abs(field_profile))
        plt.xlabel("x")
        plt.ylabel("intensity")
        plt.show()
    

class Simulation3d:
    def __init__(self, dmd:Dmd3d, phi, theta, wavelength, field_dimensions: tuple, res, source_type) -> None:
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
    dmd = Dmd1d(0, 10, 1, 5, 10)
    sim = Simulation1d(dmd, 90, 1, (10, 10), 1, "spherical")
    print(sim.get_phase_shift_point_source(1, 0))


if __name__ == "__main__":
    main()
