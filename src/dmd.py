import numpy as np
import matplotlib.pyplot as plt
# from numba import int32, float32, float64
# from numba.experimental import jitclass

# spec = [
#     ("tilt_angle_deg", int32),
#     ("tilt_angle_rad", float64),
#     ("mirror_size", float64),
#     ("mirror_gap", float64),
#     ("mirror_coords_x", float64[:]),
#     ("mirror_coords_y", float64[:]),
#     ("nr_x", int32),
#     ("nr_sources_per_mirror", int32),
#     ("nr_sources_total", int32),
#     ("width", float64),
# ]

# @jitclass(spec)
class Dmd1d:
    def __init__(self, tilt_angle, mirror_size, mirror_gap, nr_x, nr_sources_per_mirror) -> None:
        self.tilt_angle_deg = tilt_angle
        self.tilt_angle_rad = np.deg2rad(tilt_angle)
        self.mirror_size = mirror_size
        self.mirror_gap = mirror_gap
        self.mirror_coords_x = np.linspace(0, mirror_size, nr_sources_per_mirror)
        self.mirror_coords_y = np.zeros(nr_sources_per_mirror)
        self.nr_x = nr_x
        self.nr_sources_per_mirror = nr_sources_per_mirror
        self.nr_sources_total = nr_x * nr_sources_per_mirror
        self.width = (mirror_size + mirror_gap) * nr_x - mirror_gap

    # check that values don't pass the boundaries
    def check_values(self, nr_x, s):
        if (s < 0) or (s > self.mirror_size): raise ValueError(f"Parameter s has to be inside [0, {self.mirror_size}], but value is {s}.")
        if (nr_x < 0) or (nr_x >= self.nr_x): raise ValueError(f"Parameter nr_x has to be inside [0, {self.nr_x}), but value is {nr_x}.")
    
    # calculate rotated x coordinate
    def get_x(self, nr_x, s):
        self.check_values(nr_x, s)
        cos = np.cos(self.tilt_angle_rad)
        mirror_edge = (self.mirror_size + self.mirror_gap) * nr_x - self.width / 2
        mirror_middle = self.mirror_size / 2.0
        x = mirror_edge + (s - mirror_middle) * cos + mirror_middle
        return x

    # calculate rotated y coordinate
    def get_y(self, nr_x, s):
        self.check_values(nr_x, s)
        sin = np.sin(self.tilt_angle_rad)
        mirror_middle = self.mirror_size / 2.0
        y = (s - mirror_middle) * sin
        return y
    
    def get_phase_shift_old(self, nr_x, s, k, incident_angle_rad):
        return k * (self.get_x(nr_x, s) * np.cos(incident_angle_rad) + self.get_y(nr_x, s) * np.sin(incident_angle_rad))
    
    def get_projection(self, nr_x, k, incident_angle_rad):
        r_p = np.array([
            self.get_x(nr_x, self.mirror_coords_x[0]) - self.get_x(nr_x, self.mirror_coords_x[-1]),
            self.get_y(nr_x, self.mirror_coords_x[0]) - self.get_y(nr_x, self.mirror_coords_x[-1])])
        r_p_norm = r_p/np.linalg.norm(r_p)
        k_vector = k * np.array([np.cos(incident_angle_rad), np.sin(incident_angle_rad)])
        return np.dot(k_vector, r_p_norm) * r_p_norm
    
    def get_phase_shift(self, nr_x, s, k_proj):
        r = np.array([self.get_x(nr_x, s), self.get_y(nr_x, s)])
        return np.dot(k_proj, r)
    
    def display_dmd(self):
        

        for nr_x in range(self.nr_x):
            plt.plot([self.get_x(nr_x, s) for s in self.mirror_coords_x], [self.get_y(nr_x, s) for s in self.mirror_coords_x])


        plt.title("Surface of the dmd")
        plt.axis("equal")
        plt.axhline(0, linestyle="dotted", zorder=-1, color="gray")
        plt.tight_layout()
        plt.show()


class Dmd:
    """
    Class for creating a DMD consisting of an array of flat
    mirrors which are able to rotate along their diagonal
    axis.
    """

    def __init__(self, nr_x, nr_y, mirror_size, gap) -> None:
        """
        Constructor for DMD
        """
        self.nr_x = nr_x  # amount of mirrors in the x direction
        self.nr_y = nr_y  # amount of mirrors in the y direction
        self.mirror_width = mirror_size  # width of the mirrors
        self.mirror_height = mirror_size  # height of the mirrors
        self.gap_x = gap  # gap between mirrors in the x direction
        self.gap_y = gap  # gap between mirrors in the y direction
        self.dmd_width = nr_x * (mirror_size + gap) - gap  # total width of the dmd (x direction)
        self.dmd_height = nr_y * (mirror_size + gap) - gap  # total height of the dmd (y direction)

    def range_check(self, m_x, m_y, tilt_angle, s, t):
        """
        Check if parameters of mirror are within the defined range of the dmd.
        """
        if m_x < 0 or m_x >= self.nr_x:
            print("m_x has to be between 0 and {}".format(self.nr_x))
        if m_y < 0 or m_y >= self.nr_y:
            print("m_y has to be between 0 and {}".format(self.nr_y))
        if np.abs(tilt_angle) > 90:
            print("Tilt angle has to be inside [-90, 90]")
        if s < 0 or s > self.mirror_width:
            print("s has to be inside [0, {}]".format(self.mirror_width))
        if t < 0 or t > self.mirror_width:
            print("t has to be inside [0, {}]".format(self.mirror_height))

    def get_x(self, nr_x, nr_y, tilt_angle, s, t):
        """
        Calculate the x coordinates depending on mirror properties
        """
        self.range_check(nr_x, nr_y, tilt_angle, s, t)
        gamma = (2 * np.pi * tilt_angle) / 360  # in radians
        cos = np.cos(gamma)
        matrix_product = s * (0.5 * (1 - cos) + cos) + t * (0.5 * (1 - cos))
        return matrix_product + (self.mirror_width + self.gap_x) * nr_x - self.dmd_width / 2

    def get_y(self, nr_x, nr_y, tilt_angle, s, t):
        """
        Calculate the y coordinates depending on mirror properties
        """
        self.range_check(nr_x, nr_y, tilt_angle, s, t)
        gamma = (2 * np.pi * tilt_angle) / 360  # in radians
        cos = np.cos(gamma)
        matrix_product = s * (0.5 * (1 - cos)) + t * (0.5 * (1 - cos) + cos)
        return matrix_product + (self.mirror_width + self.gap_y) * nr_y - self.dmd_height / 2

    def get_z(self, nr_x, nr_y, tilt_angle, s, t):
        """
        Calculate the z coordinates depending on mirror properties
        """
        self.range_check(nr_x, nr_y, tilt_angle, s, t)
        gamma = (2 * np.pi * tilt_angle) / 360  # in radians
        sin = np.sin(gamma)
        matrix_product = 1 / np.sqrt(2) * (- s * sin + t * sin)
        return matrix_product

    def get_coordinates(self, nr_x: int, nr_y: int, tilt_angle: float, s: float, t: float):
        x = self.get_x(nr_x, nr_y, tilt_angle, s, t)
        y = self.get_y(nr_x, nr_y, tilt_angle, s, t)
        z = self.get_z(nr_x, nr_y, tilt_angle, s, t)
        return np.array([x, y, z])

    def get_surface_view(self, pixels_per_micron, tilt_angles_deg):
        st_per_mirror = pixels_per_micron * 10
        offset_x = self.dmd_width / 2
        offset_y = self.dmd_height / 2
        offset_z = 0
        width = int(self.nr_x * (self.mirror_width + self.gap_x) * pixels_per_micron)
        height = int(self.nr_y * (self.mirror_height + self.gap_y) * pixels_per_micron)
        surface = np.zeros((width, height))

        for mx in range(self.nr_x):
            for my in range(self.nr_y):
                for ss in range(st_per_mirror):
                    s = self.mirror_width / st_per_mirror * ss
                    for tt in range(st_per_mirror):
                        t = self.mirror_height / st_per_mirror * tt
                        x = self.get_x(mx, my, tilt_angles_deg[mx, my], s, t)
                        y = self.get_y(mx, my, tilt_angles_deg[mx, my], s, t)
                        z = self.get_z(mx, my, tilt_angles_deg[mx, my], s, t)

                        set_x = int((x + offset_x) * pixels_per_micron)
                        set_y = int((y + offset_y) * pixels_per_micron)
                        set_z = float(z + offset_z)
                        try:
                            surface[set_x, set_y] = set_z
                        except IndexError:
                            pass  # if calculated x|y value is out of the image boundaries, the z value will be ignored

        return surface

def save_surface(surface, path, title, x_label, y_label, z_label):
    plt.figure(figsize=(10, 8))
    plt.imshow(surface, cmap='viridis', origin='lower')
    plt.colorbar(label=z_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(path, format='pdf')


def main():
    dmd = Dmd(nr_x=20, nr_y=20, mirror_size=10, gap=0.5)
    surface = dmd.get_surface_view(5, np.ones(shape=(20, 20)) * 12)
    save_surface(surface, "../out/test_image.pdf", "surface visualization", "x", "y", "z")


if __name__ == "__main__":
    main()
