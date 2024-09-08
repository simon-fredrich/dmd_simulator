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
        
    