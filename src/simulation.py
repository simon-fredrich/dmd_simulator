import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
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

        # phase shift accross mirror
        self.phase_accros_mirror=np.zeros((self.dmd.nr_s, self.dmd.nr_s), dtype=float)
        self.phase_accros_dmd=np.zeros((self.dmd.nr_m, self.dmd.nr_m), dtype=float)

        logging.info(meta.output())

    def compute_phase_accross_mirror(self, si, sj, tilt_state) -> np.ndarray:
        # x0, y0, z0=self.dmd.grid[0, 0, 0], self.dmd.grid[0, 0, 1], 0
        x0, y0, z0=0, 0, 0
        
        if tilt_state==1:
            xi, yi, zi=self.dmd.on_positions[0][si, sj], self.dmd.on_positions[1][si, sj], self.dmd.on_positions[2][si, sj]
        elif tilt_state==0:
            xi, yi, zi=self.dmd.off_positions[0][si, sj], self.dmd.off_positions[1][si, sj], self.dmd.off_positions[2][si, sj]
        else:
            raise ValueError("Tilt state must be 0 ('off') or 1 ('on').")

        return ( self.k_wave[0] * (xi - x0) + self.k_wave[1] * (yi - y0) + self.k_wave[2] * (zi - z0) )%(2*np.pi)

    def compute_phase_accross_dmd(self, mi, mj) -> np.ndarray:
        dx=self.dmd.grid[mi, mj, 0]  # distance of x-position of mirror to origin
        dy=self.dmd.grid[mi, mj, 1]  # distance of y-position of mirror to origin
        return np.dot(self.k_wave, np.array([dx, dy, 0]))%(2*np.pi)
    
    def show_phase_accross_mirror(self, tilt_state) -> None:
        phase_accross_mirror = np.array([
            self.compute_phase_accross_mirror(si, sj, tilt_state)
            for sj in range(self.dmd.nr_s)
            for si in range(self.dmd.nr_s)
        ]).reshape(self.dmd.nr_s, self.dmd.nr_s)
        x_coords=np.linspace(0, self.dmd.m_size, self.dmd.nr_s)
        y_coords=np.linspace(0, self.dmd.m_size, self.dmd.nr_s)
        X, Y=np.meshgrid(x_coords, y_coords)

        mesh=plt.pcolormesh(X, Y, phase_accross_mirror, shading='auto', cmap='viridis')
        # Add the colorbar
        cbar = plt.colorbar(mesh)

        # Set the label for the colorbar
        cbar.set_label("Phasenversatz (radians)")

        # Set the ticks on the colorbar to multiples of pi
        cbar_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        cbar.set_ticks(cbar_ticks)
        
        # Format the ticks as multiples of pi, correctly showing 3π/2 as '3π/2'
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val, pos: (
                r"$0$" if val == 0 else
                r"$\frac{\pi}{2}$" if np.isclose(val, np.pi/2) else
                r"$\pi$" if np.isclose(val, np.pi) else
                r"$\frac{3\pi}{2}$" if np.isclose(val, 3*np.pi/2) else
                r"$2\pi$"
            )
        ))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        plt.show()

    def get_dmd_phase(self) -> np.ndarray:
        # Compute phase shifts for the "on" and "off" states across the mirror
        on_phase_accross_mirror = np.array([
            self.compute_phase_accross_mirror(si, sj, 1)
            for sj in range(self.dmd.nr_s)
            for si in range(self.dmd.nr_s)
        ]).reshape(self.dmd.nr_s, self.dmd.nr_s)

        off_phase_accross_mirror = np.array([
            self.compute_phase_accross_mirror(si, sj, 0)
            for sj in range(self.dmd.nr_s)
            for si in range(self.dmd.nr_s)
        ]).reshape(self.dmd.nr_s, self.dmd.nr_s)
        tiles=[]
        for mi in range(self.dmd.nr_m):
            row_tiles=[]
            for mj in range(self.dmd.nr_m):
                if self.pattern[mi, mj]==0:
                    row_tiles.append(off_phase_accross_mirror+self.compute_phase_accross_dmd(mi, mj))
                elif self.pattern[mi, mj]==1:
                    row_tiles.append(on_phase_accross_mirror+self.compute_phase_accross_dmd(mi, mj))
                else:
                    return ValueError("pattern values must be 0 ('off') or 1 ('on').")
            tiles.append(row_tiles)
        return np.block(tiles)


    def show_phase_accross_dmd(self) -> None:
        dmd_phase=self.get_dmd_phase()
        x_coords=np.linspace(0, self.dmd.d_size, self.dmd.nr_m*self.dmd.nr_s)
        y_coords=np.linspace(0, self.dmd.d_size, self.dmd.nr_m*self.dmd.nr_s)
        X, Y=np.meshgrid(x_coords, y_coords)

        plt.figure(figsize=(12, 12))
        mesh=plt.pcolormesh(X, Y, dmd_phase%(2*np.pi), shading='auto', cmap='viridis')

        # Add the colorbar
        cbar = plt.colorbar(mesh)

        # Set the label for the colorbar
        cbar.set_label("Phasenversatz (radians)")

        # Set the ticks on the colorbar to multiples of pi
        cbar_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        cbar.set_ticks(cbar_ticks)
        
        # Format the ticks as multiples of pi, correctly showing 3π/2 as '3π/2'
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val, pos: (
                r"$0$" if val == 0 else
                r"$\frac{\pi}{2}$" if np.isclose(val, np.pi/2) else
                r"$\pi$" if np.isclose(val, np.pi) else
                r"$\frac{3\pi}{2}$" if np.isclose(val, 3*np.pi/2) else
                r"$2\pi$"
            )
        ))

        plt.axis('equal')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Phasenversatz auf der DMD-Oberfläche")
        plt.show()

    def show_height_accross_mirror(self, tilt_state) -> None:
        if tilt_state == 0:
            Z=self.dmd.off_positions[2]
        elif tilt_state == 1:
            Z=self.dmd.on_positions[2]
        else:
            raise ValueError("Tilt state must be 0 ('off') or 1 ('on').")

        x_coords=np.linspace(0, self.dmd.m_size, self.dmd.nr_s)
        y_coords=np.linspace(0, self.dmd.m_size, self.dmd.nr_s)
        X, Y=np.meshgrid(x_coords, y_coords)

        plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
        plt.colorbar(label="Height")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        plt.show()

    def get_dmd_height(self) -> np.ndarray:
        tiles=[]
        for mi in range(self.dmd.nr_m):
            row_tiles=[]
            for mj in range(self.dmd.nr_m):
                if self.pattern[mi, mj]==0:
                    row_tiles.append(self.dmd.off_positions[2])
                elif self.pattern[mi, mj]==1:
                    row_tiles.append(self.dmd.on_positions[2])
                else:
                    return ValueError("pattern values must be 0 ('off') or 1 ('on').")
            tiles.append(row_tiles)
        return np.block(tiles)

    def show_height_accross_dmd(self) -> None:
        dmd_height = self.get_dmd_height()
        print(type(dmd_height))


        # Use the sparse option in np.meshgrid or just use x_coords and y_coords directly
        x_coords = np.linspace(0, self.dmd.d_size, self.dmd.nr_m * self.dmd.nr_s)
        y_coords = np.linspace(0, self.dmd.d_size, self.dmd.nr_m * self.dmd.nr_s)

        # Plot the height map
        plt.figure(figsize=(12, 12))
        plt.pcolormesh(x_coords, y_coords, dmd_height, shading='auto', cmap='viridis')
        plt.colorbar(label="Height")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        plt.show()

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
        initial_field = ComplexField(screen)
        point_source_pos = self.dmd.on_positions if tilt_state==1 else self.dmd.off_positions
        total_points = len(point_source_pos[0].flatten())
        current_points=0

        logging.info(f"Computing initial field for mirror at grid-position ({mi}, {mj}) with tilt state {tilt_state}...")

        for si in range(self.dmd.nr_s):
            for sj in range(self.dmd.nr_s):
                phase_shift=self.compute_phase_accross_mirror(si, sj, tilt_state)
                r = np.sqrt(np.square(screen.X - self.dmd.on_positions[0][si, sj])+
                            np.square(screen.Y - self.dmd.on_positions[1][si, sj])+
                            np.square(screen.Z - self.dmd.on_positions[2][si, sj]))
                initial_field.add(np.exp(1j * (self.k * r + phase_shift)) / r)

                # Log progress every 10% of completion
                if (current_points + 1) % (total_points // 10) == 0 or current_points == total_points - 1:
                    logging.info(f"current_points: {(current_points + 1) / total_points:.0%} complete")

                current_points+=1

        logging.info(f"Finished computing initial field for mirror at grid-position ({mi}, {mj}) with tilt state {tilt_state}.\n")

        return initial_field
    
    def compute_mirror_contribution(self, mi, mj, initial_field) -> ComplexField:
        mirror_phase=self.compute_phase_accross_dmd(mi, mj)
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

