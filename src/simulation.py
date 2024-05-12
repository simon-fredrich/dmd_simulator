import numpy as np
from dmd import Dmd, save_surface


class Simulation:
    def __init__(self, dmd, lam, t_min, t_max, t_step_size, p_min, p_max, p_step_size, tilt_angle):
        self.dmd = dmd
        self.in_beam = np.zeros(2)
        self.lam = lam
        self.t_min = t_min
        self.t_max = t_max
        self.p_min = p_min
        self.p_max = p_max
        self.p_step_size = p_step_size
        self.t_step_size = t_step_size
        self.t_values = np.linspace(self.t_min + self.t_step_size, self.t_max, self.t_step_size)
        self.p_values = np.linspace(self.p_min + self.p_step_size, self.p_max, self.t_step_size)
        self.tilt_angle = tilt_angle
        self.out_angles = self.calc_spherical_out_angles()
        self.tilt_states = np.full((self.dmd.nr_x, self.dmd.nr_y), True)

    def set_in_beam(self, in_beam):
        self.in_beam = in_beam

    def calc_spherical_out_angles(self):
        angles = np.empty((self.t_max, self.p_max), dtype=object)
        for th in range(self.t_max):
            theta = self.t_min + th * self.t_step_size
            for ph in range(self.p_max):
                phi = self.p_min + ph * self.p_step_size
                angles[th, ph] = np.array([phi, theta])
        return angles

    def calc_field_single_mirror(self, tilt_state):
        field = np.zeros((self.t_max, self.p_max), dtype=complex)
        gamma = self.tilt_angle * 2 * np.pi / 360
        gamma = gamma if tilt_state else -gamma

        for th in range(self.t_max):
            for ph in range(self.p_max):
                field[th, ph] = self.calc_field_single_out_angle(self.out_angles[th, ph], gamma)

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
        true_intensity = self.build_intensity_image(mirror_true)
        return true_intensity

def main():
    dmd = Dmd(nr_x=10, nr_y=10, mirror_size=10, gap=0.5)
    sim = Simulation(dmd, 600e-1, -90, 90, 3, -90, 90, 0.5, 12)
    theta_in = 45
    phi_in = 45
    in_beam = np.array([phi_in * 2 * np.pi / 360, theta_in * 2 * np.pi / 360])
    intensity = sim.simulate_phase_shifting(in_beam)
    save_surface(intensity, "../out/intensity.pdf", "intensity distribution", "Phi", "Theta", "Intensity")
    print(intensity)


if __name__ == "__main__":
    main()
