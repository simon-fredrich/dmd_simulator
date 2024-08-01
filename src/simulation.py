import numpy as np
from matplotlib import pyplot as plt
from dmd import Dmd2d, Dmd3d
import time
from scipy.fftpack import fft, fftfreq, fft2, ifft2, fftshift, ifftshift
from joblib import Parallel, delayed

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
    def __init__(self, dmd:Dmd3d, incident_angle_deg, wavelength, if_phase_shift=True) -> None:
        self.dmd = dmd
        self.phase_origin = np.array([-np.sqrt(2)/2*dmd.d_size, 0, 0])

        # incident wave parameters
        self.k = 2*np.pi/wavelength
        self.incident_angle_deg = incident_angle_deg
        self.incident_angle_rad = np.deg2rad(incident_angle_deg)

        # 3d vector in xz-plane
        self.k_wave = - self.k * np.array([
            np.sin(self.incident_angle_rad),
            0,
            np.cos(self.incident_angle_rad)
        ])

        self.if_phase_shift = if_phase_shift
        self.points = []
        self.phases = []

    def get_quality(self, dim, res):
        pixels_x = int(res * dim[0])
        pixels_y = int(res * dim[1])
        pixels_z = int(res * dim[2])
        return pixels_x, pixels_y, pixels_z

    def get_E_incident(self, width, height, res=1):
        dim = (width, 0, height)
        pixels_x, _, pixels_z = self.get_quality(dim, res)
        
        # define the range where the field should be calculated
        x_range = np.linspace(-dim[0]/2, dim[0]/2, pixels_x)
        z_range = np.linspace(0, dim[2], pixels_z) - self.dmd.m_size/2
        X, Z = np.meshgrid(x_range, z_range)

        # calculation of incident field
        E_incident = np.exp(1j * self.k * (X*np.sin(self.incident_angle_rad) + Z*np.cos(self.incident_angle_rad)))
        return E_incident
    
    def compute_phase_shift_and_contribution(self, mi, mj, X, Y, Z, epsilon0):
        offset_x, offset_y = self.dmd.grid[mi, mj, 0], self.dmd.grid[mi, mj, 1]

        D3 = np.dot(self.dmd.rot_matrix_y(self.dmd.pattern[mi, mj]*self.dmd.tilt_angle_rad), self.dmd.rot_matrix_z(self.dmd.rot_rad_z))
        x_rot, y_rot, z_rot = np.dot(D3, np.vstack([self.dmd.X.flatten(), self.dmd.Y.flatten(), self.dmd.Z.flatten()]))

        X_rot = x_rot.reshape(self.dmd.X.shape) - offset_x
        Y_rot = y_rot.reshape(self.dmd.Y.shape) - offset_y
        Z_rot = z_rot.reshape(self.dmd.Z.shape)

        E_total_local = np.zeros_like(X, dtype=complex)
        points_local = []
        phases_local = []

        for si in range(self.dmd.nr_s):
            for sj in range(self.dmd.nr_s):
                # changing the y value of vector to point source
                p = np.array([X_rot[si, sj], Y_rot[si, sj], Z_rot[si, sj]]) - self.phase_origin
                p_abs = np.linalg.norm(p)
                k_p = np.dot(self.k_wave, p) * p / np.square(p_abs)
                phase_shift = np.dot(k_p, p) % (2 * np.pi)

                points_local.append([X_rot[si, sj], Y_rot[si, sj], Z_rot[si, sj]])
                phases_local.append(phase_shift)

                r = np.sqrt(np.square(X - X_rot[si, sj]) + np.square(Y - Y_rot[si, sj]) + np.square(Z - Z_rot[si, sj]))
                E_total_local += np.exp(1j * (self.k * r + phase_shift)) / (r + epsilon0)

        return E_total_local, points_local, phases_local
    
    def get_E_reflected(self, width=0, depth=0, height=0, res=1, cr_plane="xz", y_value=0, z_value=0, source_type="spherical"):
        pixels_x, pixels_y, pixels_z = self.get_quality((width, depth, height), res)

        # define the range where the field should be calculated
        x_range = np.zeros((pixels_x))
        y_range = np.zeros((pixels_y))
        z_range = np.zeros((pixels_z))
        X, Y, Z = None, None, None

        # determine field coordinates
        if cr_plane=="xz":
            x_range = np.linspace(-width/2, width/2, pixels_x)
            z_range = np.linspace(0, height, pixels_z) - self.dmd.m_size
            X, Z = np.meshgrid(x_range, z_range)
            Y = np.ones_like(X) * (y_value)
        elif cr_plane=="xy":
            x_range = np.linspace(-width/2, width/2, pixels_x)
            y_range = np.linspace(-depth/2, depth/2, pixels_y)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.ones_like(X) * (z_value)
        else: ValueError(cr_plane + " is not valid. Try 'xy' or 'xz'.")

        epsilon0 = 1e-10
        E_total = np.zeros_like(X, dtype=complex)
        results = Parallel(n_jobs=8, verbose=10)(delayed(self.compute_phase_shift_and_contribution)(mi, mj, X, Y, Z, epsilon0) for mi in range(self.dmd.nr_m) for mj in range(self.dmd.nr_m))
        E_total_local, points_local, phases_local = zip(*results)
        for E in E_total_local: E_total += E
        return E_total, x_range, y_range, z_range
    
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


def show_intensities(intensities):
    fig, ax = plt.subplots(len(intensities), 1)
    for idx, intensity in enumerate(intensities):
        ax[idx].imshow(intensity, cmap='viridis', origin='lower')
        ax[idx].colorbar(label="z")
        ax[idx].xlabel("x")
        ax[idx].ylabel("y")


def main():
    import numpy as np

    # initiate dmd
    dmd = Dmd3d(
        tilt_angle_deg=12,
        m_size=5.4, 
        m_gap=5.4e-1, 
        nr_m=5,
        nr_s=20,
        pattern=np.ones((5, 5))
    )

    sim = Simulation3d(
        dmd=dmd,
        incident_angle_deg=-24,
        wavelength=532e-3,
        if_phase_shift=True
    )

    E_total_xy, X, Y, Z = sim.get_E_reflected(cr_plane="xy", width=100, depth=100, z_value=30, res=4)
    E_fft, kx, ky = sim.get_fft(E_total_xy, X, Y)

    import plotly.express as px

    # Create the heatmap with Plotly Express
    fig = px.imshow(
        E_fft,                   # Magnitude values for the heatmap
        x=np.linspace(-np.min(kx), np.max(kx), 100),                    # Spatial frequency values for x-axis
        y=np.linspace(-np.min(ky), np.max(ky), 100),                    # Spatial frequency values for y-axis
        color_continuous_scale='Viridis',  # Color scale
        labels={'x': 'Spatial Frequency (kx)', 'y': 'Spatial Frequency (ky)', 'color': 'Magnitude'},  # Labels
        title="FFT Magnitude",
        width=800,
        height=800
    )

    # Show the figure
    fig.show()



if __name__ == "__main__":
    main()
