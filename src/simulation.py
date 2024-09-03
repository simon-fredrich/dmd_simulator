import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from matplotlib import pyplot as plt
from dmd import Dmd2d, Dmd3d
from metadata import MetaData
from screen import Screen
from complex_field import ComplexField
import time
# from joblib import Parallel, delayed
# import itertools
import logging
import multiprocessing
from multiprocessing import Pool

def configure_logging():
    logger = multiprocessing.get_logger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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
    def __init__(self, dmd:Dmd3d, meta:MetaData, if_phase_shift=True) -> None:
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

        # define variables to save initial fields 
        self.initial_field_on: ComplexField
        self.initial_field_off: ComplexField

        logging.info(meta.output())

    def init_tilt_state_fields(self, screen) -> None:
        configure_logging()  # Ensure logging is configured
        m_index = self.dmd.nr_m // 2

        if np.all(self.pattern == 1):
            logging.info("Calculate initial field for 'on' state.")
            self.initial_field_on = self.compute_initial_field(screen, m_index, m_index, 1)
        elif np.all(self.pattern == 0):
            logging.info("Calculate initial field for 'off' state.")
            self.initial_field_off = self.compute_initial_field(screen, m_index, m_index, -1)
        else:
            logging.info("Calculate initial fields for 'on' & 'off' state.")

            # with Pool(processes=2) as pool:
            #     results = pool.starmap(self.compute_initial_field, [
            #         (screen, m_index, m_index, 1),
            #         (screen, m_index, m_index, -1)
            #     ])
            #     self.initial_field_on, self.initial_field_off = results

            self.initial_field_on, self.initial_field_off=\
                self.compute_initial_field(screen, m_index, m_index, 1),\
                self.compute_initial_field(screen, m_index, m_index, -1)

    def compute_field(self, pixels:int, x_min:float, x_max:float, y_min:float, y_max:float, z: float) -> ComplexField:
        screen=Screen(pixels, x_min, x_max, y_min, y_max, z)
        total_field=ComplexField(screen)

        # Compute initial fields based on pattern/hologram
        self.init_tilt_state_fields(screen)
        total_mirrors = self.dmd.nr_m*self.dmd.nr_m
        counter=0
        
        logging.info(f"Shifting initial field over {self.dmd.nr_m}x{self.dmd.nr_m} mirror-grid to construct total field.")
        for mi in range(self.dmd.nr_m):
            for mj in range(self.dmd.nr_m):
                grid_x, grid_y=self.dmd.grid[mi, mj, 0], self.dmd.grid[mi, mj, 1]
                mirror_field=self.compute_mirror_contribution(mi, mj, self.initial_field_on) if self.pattern[mi, mj]==1 else\
                    self.compute_mirror_contribution(mi, mj, self.initial_field_off)
                mirror_field.shift(grid_x, grid_y)
                total_field.mesh+=mirror_field.mesh

                # Log progress every 10% of completion
                if total_mirrors>=10:
                    if (counter + 1) % (total_mirrors // 10) == 0 or counter == total_mirrors - 1:
                        logging.info(f"Progress: {(counter + 1) / total_mirrors:.0%} complete")
                    counter+=1

        logging.info("Complete shifting.\n")


        return total_field

    def compute_initial_field(self, screen:Screen, mi, mj, tilt_state) -> ComplexField:
        # initial_field = ComplexField(screen)
        # source_pos = self.dmd.compute_position(mi, mj, tilt_state)
        # x0, y0, z0=0, 0, 0
        # for idx, (xi, yi, zi) in enumerate(zip(source_pos[0].flatten(), source_pos[1].flatten(), source_pos[2].flatten())):
        #     if idx==0:
        #         x0, y0, z0=xi, yi, zi
        #     phase_shift=self.k_wave[0]*(xi-x0)+self.k_wave[1]*(yi-y0)+self.k_wave[2]*(zi-z0)
        #     r=np.sqrt(np.square(screen.X-xi) + np.square(screen.Y-yi) + np.square(screen.Z-zi))
        #     initial_field.add(np.exp(1j * (self.k*r + phase_shift))/r)

        # initial_field = ComplexField(screen)
        # source_pos = self.dmd.compute_position(mi, mj, tilt_state)
        # x0, y0, z0 = source_pos[:, 0, 0]
        # phase_shift = (self.k_wave[0] * (source_pos[0] - x0) +
        #             self.k_wave[1] * (source_pos[1] - y0) +
        #             self.k_wave[2] * (source_pos[2] - z0))
        # r = np.sqrt((screen.X - source_pos[0])**2 +
        #             (screen.Y - source_pos[1])**2 +
        #             (screen.Z - source_pos[2])**2)
        # initial_field.add(np.exp(1j * (self.k * r + phase_shift)) / r)

        initial_field = ComplexField(screen)
        source_pos = self.dmd.compute_position(mi, mj, tilt_state)
        # x0, y0 = self.dmd.grid[self.dmd.nr_m//2, self.dmd.nr_m//2]
        x0, y0, z0=self.dmd.grid[mi, mj, 0], self.dmd.grid[mi, mj, 1], 0
        total_points = len(source_pos[0].flatten())

        logging.info(f"Computing initial field for mirror at grid-position ({mi}, {mj}) with tilt state {tilt_state}...")

        for idx, (xi, yi, zi) in enumerate(zip(source_pos[0].flatten(), source_pos[1].flatten(), source_pos[2].flatten())):
            # if idx == 0:
            #     x0, y0, z0 = xi, yi, zi
            phase_shift = self.k_wave[0] * (xi - x0) + self.k_wave[1] * (yi - y0) + self.k_wave[2] * (zi - z0)
            r = np.sqrt(np.square(screen.X - xi) + np.square(screen.Y - yi) + np.square(screen.Z - zi))
            initial_field.add(np.exp(1j * (self.k * r + phase_shift)) / r)

            # Log progress every 10% of completion
            if (idx + 1) % (total_points // 10) == 0 or idx == total_points - 1:
                logging.info(f"Progress: {(idx + 1) / total_points:.0%} complete")

        logging.info(f"Finished computing initial field for mirror at grid-position ({mi}, {mj}) with tilt state {tilt_state}.\n")

        return initial_field
    
    def compute_mirror_contribution(self, mi, mj, initial_field) -> ComplexField:
        kx=self.k_wave[0]  # projection of wavevector onto x-axis
        ky=self.k_wave[1]  # projection of wavevector onto y-axis
        dx=self.dmd.grid[mi, mj, 0]  #- self.dmd.grid[self.dmd.nr_m//2, self.dmd.nr_m//2, 0]  # distance of x-position of mirror to origin
        dy=self.dmd.grid[mi, mj, 1]  #- self.dmd.grid[self.dmd.nr_m//2, self.dmd.nr_m//2, 1]  # distance of y-position of mirror to origin
        mirror_phase=np.dot(self.k_wave, np.array([dx, dy, 0]))%(2*np.pi)

        phase_shifted_field=initial_field.copy()
        phase_shifted_field.multiply(np.exp(1j*mirror_phase))

        return phase_shifted_field
    
    def get_quality(self, dim, res):
        pixels_x = int(res * dim[0])
        pixels_y = int(res * dim[1])
        pixels_z = int(res * dim[2])
        return pixels_x, pixels_y, pixels_z
    
    def get_E_reflected(self, pixels:int, x_min:float, x_max:float, y_min:float, y_max:float, z: float) -> ComplexField:
        screen=Screen(pixels, x_min, x_max, y_min, y_max, z)

        epsilon0 = 1e-10
        E_total = ComplexField(screen)#np.zeros_like(X, dtype=complex)
        phase_origin = np.array([-np.sqrt(2)/2*self.dmd.d_size, 0, 0])
        for mi in range(self.dmd.nr_m):
            for mj in range(self.dmd.nr_m):
                offset_x, offset_y = self.dmd.grid[mi, mj, 0], self.dmd.grid[mi, mj, 1]

                # shift mirror to position on grid
                X_mirror = self.dmd.X - offset_x
                Y_mirror = self.dmd.Y - offset_y
                Z_mirror = self.dmd.Z
                
                for si in range(self.dmd.nr_s):
                    for sj in range(self.dmd.nr_s):
                        # calculate phase shift
                        p = np.array([X_mirror[si, sj], Y_mirror[si, sj], Z_mirror[si, sj]])-phase_origin
                        p_abs = np.linalg.norm(p)
                        k_p = np.dot(self.k_wave, p)*p/np.square(p_abs)
                        phase_shift = np.dot(k_p, p)%(2*np.pi)

                        # add source contribution to total field
                        r = np.sqrt(np.square(screen.X - X_mirror[si, sj]) + np.square(screen.Y - Y_mirror[si, sj]) + np.square(screen.Z - Z_mirror[si, sj]))
                        # TODO: source type
                        E_total.mesh += np.exp(1j * (self.k * r + phase_shift))/(r + epsilon0)
        return E_total
    
    def apply_aperture(self, field, x_range, y_range, aperture_constant):
        xx, yy=np.meshgrid(
            np.linspace(np.min(x_range), np.max(x_range), field.shape[0]),
            np.linspace(np.min(y_range), np.max(y_range), field.shape[1])
        )
        aperture = np.ones_like(xx)
        aperture[xx**2+yy**2>=aperture_constant**2]=0
        return field*aperture
    
    def display_pattern(self):
        plt.imshow(self.pattern, extent=[0, self.dmd.nr_m, 0, self.dmd.nr_m])
        plt.colorbar()

