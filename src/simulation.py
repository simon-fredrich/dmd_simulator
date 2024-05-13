import numpy as np
from dmd import Dmd, save_surface
from metadata import MetaData


class Simulation:
    def __init__(self, meta: MetaData):
        self.in_beam = np.zeros(2)

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

    def calc_spherical_out_angles(self, phi_min, nr_phi, theta_min, nr_theta, step_size):
        angles = np.empty((self.t_max, self.p_max), dtype=object)
        for th in range(self.t_max):
            theta = self.theta_min_d + th * self.out_step_size_d
            for ph in range(self.p_max):
                phi = self.phi_min_d + ph * self.out_step_size_d
                angles[th, ph] = np.array([phi, theta])
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

        az = np.sqrt(1 / (np.tan(self.in_beam[0]) ** 2 + np.tan(self.in_beam[1]) ** 2))
        ax = az * np.tan(self.in_beam[0])
        ay = az * np.tan(self.in_beam[1])

        if (np.tan(out[0]) ** 2 + np.tan(out[1]) ** 2) == 0:
            print("Durch null geteilt.")
            print(out)

        bz = np.sqrt(1 / (np.tan(out[0]) ** 2 + np.tan(out[1]) ** 2))
        bx = bz * np.tan(out[0])
        by = bz * np.tan(out[1])

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


def main():
    meta = MetaData()
    meta.out_dir = "../out"

    meta.lambdas = [631]

    meta.nr_x = 20
    meta.nr_y = 20

    meta.lattice_constant = 7.56
    meta.fill_factor = 0.92
    meta.tilt_angle = 12.0

    meta.phi_out_start = -90
    meta.phi_out_end = 90
    meta.theta_out_start = -90
    meta.theta_out_end = 90
    meta.out_step_size = 2.26

    meta.phi_in_start = 34.05
    meta.phi_in_end = 35.05
    meta.theta_in_start = -34.05
    meta.theta_in_end = -33.05
    meta.in_step_size = 1.0

    sim = Simulation(meta)
    theta_in = 23
    phi_in = 23
    in_beam = np.array([phi_in * 2 * np.pi / 360, theta_in * 2 * np.pi / 360])
    intensities = sim.simulate_phase_shifting(in_beam)
    save_surface(intensities[0], "../out/true_intensity.pdf", "intensity distribution", "Phi",
                 "Theta", "True Intensity")
    save_surface(intensities[1], "../out/false_intensity.pdf", "intensity distribution", "Phi",
                 "Theta", "False ""Intensity")


if __name__ == "__main__":
    main()
