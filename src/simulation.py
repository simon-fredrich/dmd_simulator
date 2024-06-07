import numpy as np
from matplotlib import pyplot as plt

from dmd import Dmd, Dmd1d, save_surface
from metadata import MetaData

# from numba import int32, float32, complex64, char   # import the types
# from numba.experimental import jitclass

# spec = [
#     ('dmd', Dmd1d.class_type.instance_type),
#     ('k', float32),
#     ("source_type", char),
#     ("res", int32),
#     ("pixels_x", int32),
#     ("pixels_y", int32),
#     ("incident_angle_deg", int32),
#     ("incident_angle_rad", float32),
#     ("x_range", float32),
#     ("y_range", float32),
#     ("X", float32),
#     ("Y", float32),
#     ("E_incident", complex64)
# ]

'''Below is the simulation for 1d mirrors.'''

# @jitclass(spec)
class Simulation1d:
    def __init__(self, dmd:Dmd1d, incident_angle, wavelength, field_dimensions: tuple, res, source_type) -> None:
        self.dmd = dmd
        self.k = 2 * np.pi / wavelength
        self.source_type = source_type
        self.res = res
        self.pixels_x = res * field_dimensions[0]
        self.pixels_y = res * field_dimensions[1]
        self.incident_angle_deg = incident_angle
        self.incident_angle_rad = np.deg2rad(incident_angle)

        # define the range where the field should be calculated
        self.x_range = np.linspace(-field_dimensions[0]/2, field_dimensions[0]/2, self.pixels_x)
        self.y_range = np.linspace(0, field_dimensions[1], self.pixels_y) - self.dmd.mirror_size/2
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)

        self.E_incident = np.exp(1j * self.k * (self.X * np.cos(self.incident_angle_rad) + self.Y * np.sin(self.incident_angle_rad)))

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


'''Below should be the simulation approach for 2d mirrors, but is not yet implemented correctly'''

class Simulation:
    def __init__(self, meta: MetaData):
        self.in_beam = Vector(1, 1)

        self.meta = meta
        self.nr_x = meta.nr_x
        self.nr_y = meta.nr_y
        self.mirror_size = np.sqrt(meta.lattice_constant*meta.lattice_constant*meta.fill_factor)
        self.gap = meta.lattice_constant - self.mirror_size
        self.dmd = Dmd(self.nr_x, self.nr_y, self.mirror_size, self.gap)

        self.tilt_angle = meta.tilt_angle

        self.lam = meta.lambdas[0]
        self.lambdaUm = self.lam * 0.001

        self.phi_min_d = meta.phi_out_start
        self.phi_max_d = meta.phi_out_end
        self.theta_min_d = meta.theta_out_start
        self.theta_max_d = meta.theta_out_end
        self.out_step_size_d = meta.out_step_size

        self.t_max = int((self.theta_max_d - self.theta_min_d) / self.out_step_size_d)
        self.p_max = int((self.phi_max_d - self.phi_min_d) / self.out_step_size_d)
        self.tilt_angle = meta.tilt_angle
        self.tilt_states = np.full((meta.nr_x, meta.nr_y), True)

        self.reference_position = self.dmd.get_coordinates(0, 0, self.tilt_angle, 0, 0)

        self.mirror_true = np.zeros((self.t_max, self.p_max), dtype=complex)
        self.mirror_false = np.zeros((self.t_max, self.p_max), dtype=complex)
        self.final_field = np.zeros((self.t_max, self.p_max), dtype=complex)

        self.out_angles = np.empty((self.t_max, self.p_max), dtype=object)

        self.init()

    def init(self):
        # Switching from degrees to radians
        theta_min_r = self.theta_min_d * np.pi / 180
        theta_max_r = self.theta_max_d * np.pi / 180
        phi_min_r = self.phi_min_d * np.pi / 180
        phi_max_r = self.phi_max_d * np.pi / 180
        out_step_size_r = self.out_step_size_d * np.pi / 180

        # Pre-calculations over the mirrors
        tilt_states = np.zeros((self.nr_y, self.nr_x), dtype=bool)
        dmd_positions = np.zeros((self.nr_y, self.nr_x), dtype=object)

        for my in range(self.nr_y):
            for mx in range(self.nr_x):
                tilt_states[my, mx] = True
                # angle_of_tilt = self.tilt_angle if tilt_states[my, mx] else -self.tilt_angle
                # dmd_positions[my, mx] = self.dmd.get_coordinates(mx, my, angle_of_tilt, 0, 0)

        self.out_angles = self.calc_spherical_out_angles(phi_min_r, self.p_max, theta_min_r, self.t_max,
                                                         out_step_size_r)

        print("initialization complete!")

    def set_in_beam(self, in_beam):
        self.in_beam = in_beam

    def calc_out_beam(self):
        # maximum values in the mirror coordinate system to get the corner points of the mirror
        s_max = self.dmd.mirror_width
        t_max = self.dmd.mirror_height

        # calculate three points for representing the mirror plane
        A = self.dmd.get_coordinates(0, 0, 12, 0, 0)
        B = self.dmd.get_coordinates(0, 0, 12, 0, t_max)
        C = self.dmd.get_coordinates(0, 0, 12, s_max, 0)

        # calculating vectors between one corner point and the others (of the mirror)
        AB = B - A
        AC = C - A

        # calculating the normal to the mirror plane
        n_0 = np.cross(AB, AC)

        # normalizing the normal to the mirror plane
        n = n_0 / np.linalg.norm(n_0)

        # calculating the outgoing beam as a vector
        out_beam = in_beam - 2 * np.dot(np.dot(in_beam, n), n)

        # printing the incident and outgoing beam in the terminal
        print("in_beam: ", self.in_beam)
        print("out_beam: ", out_beam)

        return out_beam

    def calc_field_at_pos(self):
        pass

    def calc_spherical_out_angles(self, phi_min, nr_phi, theta_min, nr_theta, step_size):
        angles = np.empty((self.t_max, self.p_max), dtype=object)
        for th in range(self.t_max):
            theta = self.theta_min_d + th * self.out_step_size_d
            for ph in range(self.p_max):
                phi = self.phi_min_d + ph * self.out_step_size_d
                angles[th, ph] = Vector(phi * 2 * np.pi / 360, theta * 2 * np.pi / 360)
        return angles

    def calc_field_single_mirror(self, tilt_state):
        field = np.zeros((self.t_max, self.p_max), dtype=complex)
        gamma = self.tilt_angle * 2 * np.pi / 360
        gamma = gamma if tilt_state else -gamma

        for th in range(self.t_max):
            for ph in range(self.p_max):
                field[th, ph] = self.calc_field_single_out_angle(self.out_angles[th, ph], gamma)

        print("field calculated!")

        return field

    def calc_field_single_out_angle(self, out, gamma):
        m = self.dmd.mirror_width

        ax = self.in_beam.x
        ay = self.in_beam.y
        az = self.in_beam.z

        bx = out.x
        by = out.y
        bz = out.z

        s2 = np.sqrt(2)
        cg = np.cos(gamma)
        sg = np.sin(gamma)

        r0 = self.lam * self.lam
        r1 = np.pi * np.pi
        r2 = ax + ay - bx - by + (ax - ay - bx + by) * cg - s2 * (az - bz) * sg
        r3 = -ax - ay + bx + by + (ax - ay - bx + by) * cg - s2 * (az - bz) * sg
        r = r0 / r1 / r2 / r3

        arg_factor = np.pi / self.lam
        arg0 = 0
        arg1 = (2 * ax * m + 2 * ay * m - 2 * bx * m - 2 * by * m) * arg_factor
        arg2 = (ax * m + ay * m - bx * m - by * m + (ax - ay - bx + by) * m * cg - s2 * (az - bz) * m * sg) * arg_factor
        arg3 = (ax * m + ay * m - bx * m - by * m - (ax - ay - bx + by) * m * cg + s2 * (az - bz) * m * sg) * arg_factor

        re0 = np.cos(arg0)
        im0 = np.sin(arg0)
        re1 = np.cos(arg1)
        im1 = np.sin(arg1)
        re2 = np.cos(arg2)
        im2 = np.sin(arg2)
        re3 = np.sin(arg3)
        im3 = np.sin(arg3)

        re = r * (re0 + re1 - re2 - re3)
        im = r * (im0 + im1 - im2 - im3)

        return re + 1j * im

    def build_intensity_image(self, field):
        width = len(field[0])
        height = len(field)
        view = np.zeros((width, height))
        for th in range(height):
            for ph in range(width):
                view[th, ph] = abs(field[th, ph])
        return view

    def simulate_phase_shifting(self, in_beam):
        self.set_in_beam(in_beam)
        mirror_true = self.calc_field_single_mirror(True)
        mirror_false = self.calc_field_single_mirror(False)
        true_intensity = self.build_intensity_image(mirror_true)
        false_intensity = self.build_intensity_image(mirror_false)
        return true_intensity, false_intensity

def show_intensities(intensities):
    fig, ax = plt.subplots(len(intensities), 1)
    for idx, intensity in enumerate(intensities):
        ax[idx].imshow(intensity, cmap='viridis', origin='lower')
        ax[idx].colorbar(label="z")
        ax[idx].xlabel("x")
        ax[idx].ylabel("y")


def main():
    meta = MetaData()
    meta.out_dir = "../out"

    meta.lambdas = [631]

    meta.nr_x = 100
    meta.nr_y = 100

    meta.lattice_constant = 7.56
    meta.fill_factor = 0.92
    meta.tilt_angle = 12.0

    meta.phi_out_start = -90
    meta.phi_out_end = 90
    meta.theta_out_start = -90
    meta.theta_out_end = 90
    meta.out_step_size = 2.66

    meta.phi_in_start = 34.05
    meta.phi_in_end = 35.05
    meta.theta_in_start = -34.05
    meta.theta_in_end = -33.05
    meta.in_step_size = 1.0

    sim = Simulation(meta)
    theta_in = 0.01
    phi_in = 0.01
    in_beam = Vector(phi_in * 2 * np.pi / 360, theta_in * 2 * np.pi / 360)
    intensities = sim.simulate_phase_shifting(in_beam)
    save_surface(intensities[0], "../out/true_intensity.pdf", "intensity distribution for true state", "Phi",
                 "Theta", "True Intensity")
    save_surface(intensities[1], "../out/false_intensity.pdf", "intensity distribution for false state", "Phi",
                 "Theta", "False ""Intensity")

    plt.show()


    # surface = sim.dmd.get_surface_view(100, np.ones(shape=(sim.nr_x, sim.nr_y)) * sim.tilt_angle)
    # save_surface(surface, "../out/test_image.pdf", "surface visualization", "x", "y", "z")


if __name__ == "__main__":
    main()
