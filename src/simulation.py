import numpy as np
from matplotlib import pyplot as plt
from dmd import Dmd2d, Dmd3d
from metadata import MetaData
from screen import Screen
from complex_field import ComplexField
import time
from scipy.fftpack import fft, fftfreq, fft2, ifft2, fftshift, ifftshift
from joblib import Parallel, delayed
from multiprocessing import Pool
import itertools

'''Below is the simulation for 2d mirrors.'''

class Simulation2d:
    def __init__(self, dmd:Dmd2d, incident_angle_deg, wavelength, if_phase_shift=True) -> None:
        self.dmd = dmd

        # incident wave parameters
        self.k = 2*np.pi/wavelength
        self.incident_angle_rad = np.deg2rad(incident_angle_deg)

        # 3d vector in xz-plane
        self.k_wave = - self.k * np.array([
            np.sin(self.incident_angle_rad),
            np.cos(self.incident_angle_rad)
        ])

        self.if_phase_shift = if_phase_shift
        self.points = []
        self.phases = []

    def get_quality(self, dim, res):
        pixels_x = int(res * dim[0])
        pixels_z = int(res * dim[1])
        return pixels_x, pixels_z
    
    def get_E_incident(self, width, height, res=1):
        dim = (width, height)
        pixels_x, pixels_z = self.get_quality(dim, res)
        
        # define the range where the field should be calculated
        x_range = np.linspace(-width/2, width/2, pixels_x)
        z_range = np.linspace(0, height/2, pixels_z) - self.dmd.m_size
        X, Z = np.meshgrid(x_range, z_range)

        # calculation of incident field
        E_incident = np.exp(1j * (-self.k) * (X*np.sin(self.incident_angle_rad) + Z*np.cos(self.incident_angle_rad)))
        return E_incident
    
    def get_E_reflected(self, width=0, height=0, res=1, source_type="spherical"):
        pixels_x, pixels_z = self.get_quality((width, height), res)

        # define the range where the field should be calculated
        x_range = np.linspace(-width/2, width/2, pixels_x)
        z_range = np.linspace(0, height, pixels_z) - self.dmd.m_size
        X, Z = np.meshgrid(x_range, z_range)

        epsilon0 = 1e-10
        E_total = np.zeros_like(X, dtype=complex)
        phase_origin = np.array([-self.dmd.d_size/2, 0])
        for mi in range(self.dmd.nr_m):
            offset_x = self.dmd.grid[mi]
            X_rot = self.dmd.x_rot.reshape(self.dmd.X.shape) - offset_x
            Z_rot = self.dmd.z_rot.reshape(self.dmd.Z.shape)

            for si in range(self.dmd.nr_s):
                # calculate phase shift
                p = np.array([X_rot[si, si], Z_rot[si, si]])-phase_origin
                p_abs = np.linalg.norm(p)
                k_p = np.dot(self.k_wave, p)*p/np.square(p_abs)
                phase_shift = np.dot(k_p, p)%(2*np.pi)

                # save points & phases
                self.points.append([X_rot[si, si], Z_rot[si, si]])
                self.phases.append(phase_shift)

                # add source contribution to total field
                r = np.sqrt(np.square(X - X_rot[si, si]) + np.square(Z - Z_rot[si, si]))
                # TODO: source type
                E_total += np.exp(1j * (self.k * r + phase_shift))/(r + epsilon0)
        return E_total, x_range, z_range
    
    def get_fft(self, E_total, x_range, z_range):
        X, Y = np.meshgrid(x_range, z_range)
        fft_field = fftshift(fft2(E_total))
        fft_magnitude = np.abs(fft_field)

        # Calculate the spatial frequency coordinates
        kx = np.fft.fftfreq(len(x_range), d=(X[0, 1] - X[0, 0]))
        kz = np.fft.fftfreq(len(z_range), d=(Y[1, 0] - Y[0, 0]))
        KX, KZ = np.meshgrid(kx, kz)
        angles = np.arctan2(KZ, KX)  # Angles of outgoing directions
        return fft_magnitude, kx, kz
        
    
'''Below is the simulation for 3d mirrors.'''

class Simulation3d:
    def __init__(self, dmd:Dmd3d, meta, if_phase_shift=True) -> None:
        self.dmd = dmd
        self.phase_origin = np.array([-np.sqrt(2)/2*dmd.d_size, 0, 0])
        self.pattern = meta.pattern

        # incident wave parameters
        self.k = 2*np.pi/meta.wavelength
        self.incident_angle_deg = meta.incident_angle_deg
        self.incident_angle_rad = np.deg2rad(meta.incident_angle_deg)
        self.epsilon0=1e-10

        # 3d vector in xz-plane
        self.k_wave = - self.k * np.array([
            np.sin(self.incident_angle_rad),
            0,
            np.cos(self.incident_angle_rad)
        ])

    def compute_field(self, pixels:int, x_min:float, x_max:float, y_min:float, y_max:float, z: float) -> ComplexField:
        screen=Screen(pixels, x_min, x_max, y_min, y_max, z)
        total_field=ComplexField(screen)
        initial_field=self.compute_initial_field(screen, self.dmd.nr_m//2, self.dmd.nr_m//2)
        
        for mi in range(self.dmd.nr_m):
            for mj in range(self.dmd.nr_m):
                grid_x, grid_y=self.dmd.grid[mi, mj, 0], self.dmd.grid[mi, mj, 1]
                mirror_field=self.compute_mirror_contribution(mi, mj, initial_field)
                mirror_field.shift(grid_x, grid_y)
                total_field.mesh+=mirror_field.mesh

        return total_field

    def compute_initial_field(self, screen:Screen, mi, mj) -> ComplexField:
        initial_field = ComplexField(screen)
        source_pos = self.dmd.compute_position(mi, mj, self.pattern[mi, mj])
        x0, y0, z0=0, 0, 0
        for idx, (xi, yi, zi) in enumerate(zip(source_pos[0].flatten(), source_pos[1].flatten(), source_pos[2].flatten())):
            if idx==0:
                x0, y0, z0=xi, yi, zi
            phase_shift=self.k_wave[0]*(xi-x0)+self.k_wave[1]*(yi-y0)+self.k_wave[2]*(zi-z0)
            r=np.sqrt(np.square(screen.X-xi) + np.square(screen.Y-yi) + np.square(screen.Z-zi))
            initial_field.add(np.exp(1j * (self.k*r + phase_shift))/r)

        return initial_field
    
    def compute_mirror_contribution(self, mi, mj, initial_field) -> ComplexField:
        kx=self.k_wave[0]  # projection of wavevector onto x-axis
        dx=self.dmd.grid[mi, mj, 0] - self.dmd.grid[self.dmd.nr_m//2, self.dmd.nr_m//2, 0]  # distance of x-position of mirror to origin
        dy=self.dmd.grid[mi, mj, 1] - self.dmd.grid[self.dmd.nr_m//2, self.dmd.nr_m//2, 1]  # distance of y-position of mirror to origin
        mirror_phase=kx*dx

        phase_shifted_field=initial_field.copy()
        phase_shifted_field.multiply(np.exp(1j*mirror_phase))

        return phase_shifted_field    

    def get_fft(self, E_total, x_range, y_range):
        X, Y = np.meshgrid(x_range, y_range)
        fft_field = fftshift(fft2(E_total))
        fft_magnitude = np.abs(fft_field)

        # Calculate the spatial frequency coordinates
        kx = np.fft.fftfreq(len(x_range), d=(X[0, 1] - X[0, 0]))
        ky = np.fft.fftfreq(len(y_range), d=(Y[1, 0] - Y[0, 0]))
        KX, KY = np.meshgrid(kx, ky)
        angles = np.arctan2(KY, KX)  # Angles of outgoing directions
        return fft_magnitude, kx, ky
    
    def apply_aperture(self, field, x_range, y_range, aperture_constant):
        xx, yy=np.meshgrid(
            np.linspace(np.min(x_range), np.max(x_range), field.shape[0]),
            np.linspace(np.min(y_range), np.max(y_range), field.shape[1])
        )
        aperture = np.ones_like(xx)
        aperture[xx**2+yy**2>=aperture_constant**2]=0
        return field*aperture
    