import numpy as np
from matplotlib import pyplot as plt
from dmd import Dmd2d, Dmd3d
import time
from scipy.fftpack import fft, fftfreq, fft2, ifft2, fftshift, ifftshift

'''Below is the simulation for 2d mirrors.'''

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
                phase_shift = 0#self.get_phase_shift_point_source(nr_mirror_x, nr_source)
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
    
'''Below is the simulation for 3d mirrors.'''

class Simulation3d:
    def __init__(self, dmd:Dmd3d, incident_angle, wavelength, field_dimensions: tuple, res, source_type="spherical", if_phase_shift=True) -> None:
        self.dmd = dmd

        # incident wave parameters
        self.wavelength = 1
        self.k = 2*np.pi/wavelength
        self.incident_angle_deg = - 90 + incident_angle
        self.incident_angle_rad = np.deg2rad(self.incident_angle_deg)

        # 3d vector in xz-plane
        self.k_wave = - self.k * np.array([
            - np.sin(self.incident_angle_rad),
            0,
            np.cos(self.incident_angle_rad)
        ])

        # projection of wave vector along x-axis
        self.x_unit = np.array([1, 0, 0])
        self.k_x = np.dot(self.k_wave, self.x_unit)*self.x_unit

        # projection of wave vector along mirror
        self.r_m = np.array([np.cos(dmd.tilt_angle_rad), 0, np.sin(dmd.tilt_angle_rad)])
        self.k_m = np.dot(self.k_wave, self.r_m)*self.r_m

        self.source_type = source_type
        self.if_phase_shift = if_phase_shift
        self.res = res  # resolution
        self.field_dimensions = field_dimensions
        self.pixels_x = int(res * field_dimensions[0])
        self.pixels_y = int(res * field_dimensions[1])
        self.pixels_z = int(res * field_dimensions[2])

        # array to save phase shift values
        self.phase_shifts = np.zeros((dmd.nr_mirrors, dmd.nr_mirrors, dmd.nr_sources, dmd.nr_sources))

    def get_E_incident(self):
        # define the range where the field should be calculated
        x_range = np.linspace(-self.field_dimensions[0]/2, self.field_dimensions[0]/2, self.pixels_x)
        z_range = np.linspace(0, self.field_dimensions[2], self.pixels_z) - self.dmd.mirror_size/2
        X, Z = np.meshgrid(x_range, z_range)
        Y = np.zeros_like(X)

        # calculation of incident field
        E_incident = np.exp(1j * self.k * (X*np.sin(self.incident_angle_rad) + Z*np.cos(self.incident_angle_rad)))
        return E_incident
    
    def get_phase_shift(self, sx, sy, x=0, y=0, z=0, x0=0, y0=0, z0=0):
        # calculate phase shift along x-axis
        mirror_x0 = x0[0, 0]  # x-coordinate of 0-th mirror
        mirror_x = x[sx, sy]  # x-coordinate of current mirror
        mirror_dist = np.abs(mirror_x-mirror_x0)
        gap_phase = np.dot(self.k_x, self.x_unit)*mirror_dist  # phase shift of current mirror y-row

        # calculate phase shift along mirror plane
        mirror_z = z[sx, sy]
        source_dist = np.sqrt(np.square(mirror_x - x[sx, sy])+np.square(mirror_z - z[sx, sy]))
        mirror_phase = np.dot(self.k_m, self.r_m)*source_dist  # phase shift of point source
        
        phase_shift = gap_phase + mirror_phase
        if not self.if_phase_shift: phase_shift = 0

        return phase_shift


    def get_E_total(self):
        # define the range where the field should be calculated
        x_range = np.linspace(-self.field_dimensions[0]/2, self.field_dimensions[0]/2, self.pixels_x)
        z_range = np.linspace(0, self.field_dimensions[2], self.pixels_z) - self.dmd.mirror_size/2
        X, Z = np.meshgrid(x_range, z_range)
        Y = np.zeros_like(X)

        epsilon0 = 1e-10
        E_total = np.zeros_like(X, dtype=complex)

        start = time.time()
        for mx in range(self.dmd.nr_mirrors):
            for my in range(self.dmd.nr_mirrors):
                x = self.dmd.get_x(mx, my)
                y = self.dmd.get_y(mx, my)
                z = self.dmd.get_z(mx, my)
                x0 = self.dmd.get_x(self.dmd.nr_mirrors-1, 0)

                for sx in range(self.dmd.nr_sources):
                    for sy in range(self.dmd.nr_sources):
                        # extract current coordinates
                        x_coord = x[sx, sy]
                        y_coord = y[sx, sy]
                        z_coord = z[sx, sy]

                        phase_shift = self.get_phase_shift(sx, sy, x=x, y=y, z=z, x0=x0)

                        # calculate field 
                        r = np.sqrt(np.square(X - x_coord) + np.square(Y - y_coord) + np.square(Z - z_coord))
                        E_total += np.exp(1j * (self.k * r + phase_shift))/(r + epsilon0)
                # print(mx, my)
        end = time.time()
        print(f"{end-start} seconds have passed.")

        return E_total, X, Y, Z
    
    def get_E_total_xy(self, z_value):

        z_shift = z_value + self.dmd.mirror_size/2
        # mapping the real x-coordinate to the array index
        z_mapped = int(z_shift * self.pixels_z/self.field_dimensions[2])

        # define the range where the field should be calculated
        x_range = np.linspace(-self.field_dimensions[0]/2, self.field_dimensions[0]/2, self.pixels_x)
        y_range = np.linspace(-self.field_dimensions[1]/2, self.field_dimensions[1]/2, self.pixels_y)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.ones_like(X) * z_mapped

        epsilon0 = 1e-10
        E_total = np.zeros_like(X, dtype=complex)

        start = time.time()
        for mx in range(self.dmd.nr_mirrors):
            for my in range(self.dmd.nr_mirrors):
                x = self.dmd.get_x(mx, my)
                y = self.dmd.get_y(mx, my)
                z = self.dmd.get_z(mx, my)
                x0 = self.dmd.get_x(self.dmd.nr_mirrors-1, 0)

                for sx in range(self.dmd.nr_sources):
                    for sy in range(self.dmd.nr_sources):
                        # extract current coordinates
                        x_coord = x[sx, sy]
                        y_coord = y[sx, sy]
                        z_coord = z[sx, sy]

                        phase_shift = self.get_phase_shift(sx, sy, x=x, y=y, z=z, x0=x0)

                        # calculate field 
                        r = np.sqrt(np.square(X - x_coord) + np.square(Y - y_coord) + np.square(Z - z_coord))
                        E_total += np.exp(1j * (self.k * r + phase_shift))/(r + epsilon0)
                # print(mx, my)
        end = time.time()
        print(f"{end-start} seconds have passed.")

        return E_total, X, Y, Z
    
    def get_fourier_transform(self, z_value):
        E_total_xy, X, Y, Z = self.get_E_total_xy(z_value)
        fft_field = fftshift(fft2(E_total_xy))
        fft_magnitude = np.abs(fft_field)

        # Calculate the spatial frequency coordinates
        kx = np.fft.fftfreq(self.pixels_x, d=(X[0, 1] - X[0, 0]))
        ky = np.fft.fftfreq(self.pixels_y, d=(Y[1, 0] - Y[0, 0]))
        KX, KY = np.meshgrid(kx, ky)
        angles = np.arctan2(KY, KX)  # Angles of outgoing directions

        # Plot the FFT result to analyze the outgoing directions
        plt.figure(figsize=(8, 6))
        plt.imshow(fft_magnitude, extent=(kx.min(), kx.max(), ky.min(), ky.max()), cmap='viridis', origin='lower')
        plt.colorbar(label='FFT Magnitude')
        plt.title('FFT of Outgoing Wave Field')
        plt.xlabel('Spatial Frequency kx')
        plt.ylabel('Spatial Frequency ky')
        plt.show()

        # Plot the FFT result as a function of the outgoing directions
        plt.figure(figsize=(8, 6))
        plt.polar(angles.ravel(), fft_magnitude.ravel(), '-')
        plt.title('FFT Magnitude vs. Outgoing Directions')
        plt.show()

    
    def plot_E_total(self, E_total, x, y, xlabel="x", ylabel="y"):
        # Plotting
        plt.figure(figsize=(12, 6))

        # Plot the real part of the total reflected field
        plt.contourf(x, y, np.log(np.abs(E_total)), levels=50, cmap='viridis')
        plt.title('abs(E_total)')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis("square")

        plt.tight_layout()
        plt.show()


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
