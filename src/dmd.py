import numpy as np
import matplotlib.pyplot as plt


class Dmd2d:
    def __init__(self, tilt_angle, mirror_size, mirror_gap, nr_mirrors_x, nr_sources_per_mirror) -> None:
        self.tilt_angle_deg = tilt_angle
        self.tilt_angle_rad = np.deg2rad(tilt_angle)
        self.mirror_size = mirror_size
        self.mirror_gap = mirror_gap
        self.mirror_coords_x = np.linspace(0, mirror_size, nr_sources_per_mirror)
        self.mirror_coords_y = np.zeros(nr_sources_per_mirror)
        self.nr_mirrors_x = nr_mirrors_x  # number of mirrors
        self.nr_sources_per_mirror = nr_sources_per_mirror
        self.nr_sources_total = nr_mirrors_x * nr_sources_per_mirror
        self.width = (mirror_size + mirror_gap) * nr_mirrors_x - mirror_gap

    def get_source_positions(self):
        source_positions = np.zeros((self.nr_mirrors_x, self.nr_sources_per_mirror, 2))
        for nr_mirror_x in range(self.nr_mirrors_x):
            for nr_source, s in enumerate(self.mirror_coords_x):
                source_positions[nr_mirror_x, nr_source, 0] = self.get_x(nr_mirror_x, s)
                source_positions[nr_mirror_x, nr_source, 1] = self.get_y(nr_mirror_x, s)
        return source_positions


    # check that values don't pass the boundaries
    def check_values(self, nr_mirrors_x, s):
        if (s < 0) or (s > self.mirror_size): raise ValueError(f"Parameter s has to be inside [0, {self.mirror_size}], but value is {s}.")
        if (nr_mirrors_x < 0) or (nr_mirrors_x >= self.nr_mirrors_x): raise ValueError(f"Parameter nr_mirrors_x has to be inside [0, {self.nr_mirrors_x}), but value is {nr_mirrors_x}.")
    
    # calculate rotated x coordinate
    def get_x(self, nr_mirror_x, s):
        self.check_values(nr_mirror_x, s)
        cos = np.cos(self.tilt_angle_rad)
        mirror_edge = (self.mirror_size + self.mirror_gap) * nr_mirror_x - self.width / 2.0
        mirror_middle = self.mirror_size / 2.0
        x = mirror_edge + (s - mirror_middle) * cos + mirror_middle
        return x

    # calculate rotated y coordinate
    def get_y(self, nr_mirror_x, s):
        self.check_values(nr_mirror_x, s)
        sin = np.sin(self.tilt_angle_rad)
        mirror_middle = self.mirror_size / 2.0
        y = (s - mirror_middle) * sin
        return y
    
    def display_dmd(self):
        for nr_mirrors_x in range(self.nr_mirrors_x):
            plt.plot([self.get_x(nr_mirrors_x, s) for s in self.mirror_coords_x], [self.get_y(nr_mirrors_x, s) for s in self.mirror_coords_x])
        plt.title("Surface of the dmd")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.axhline(0, linestyle="dotted", zorder=-1, color="gray")
        plt.tight_layout()
        plt.show()


class Dmd3d:
    def __init__(self, tilt_angle, mirror_size, mirror_gap, nr_mirrors_x, nr_y, nr_sources_per_x, nr_sources_per_y) -> None:
        self.tilt_angle_deg = tilt_angle
        self.tilt_angle_rad = np.deg2rad(tilt_angle)
        self.mirror_size = mirror_size
        self.mirror_gap = mirror_gap
        self.mirror_coords_x = np.linspace(0, mirror_size, nr_sources_per_x)
        self.mirror_coords_y = np.linspace(0, mirror_size, nr_sources_per_y)
        self.nr_mirrors_x = nr_mirrors_x
        self.nr_y = nr_y
        self.nr_sources_per_x = nr_sources_per_x
        self.nr_sources_per_y = nr_sources_per_y
        self.nr_sources_total = nr_mirrors_x * nr_sources_per_x + nr_y * nr_sources_per_y
        self.width = (mirror_size + mirror_gap) * nr_mirrors_x - mirror_gap
        self.height = (mirror_size + mirror_gap) * nr_y - mirror_gap

    def range_check(self, m_x, m_y, tilt_angle, s, t):
        """
        Check if parameters of mirror are within the defined range of the dmd.
        """
        if m_x < 0 or m_x >= self.nr_mirrors_x:
            print("m_x has to be between 0 and {}".format(self.nr_mirrors_x))
        if m_y < 0 or m_y >= self.nr_y:
            print("m_y has to be between 0 and {}".format(self.nr_y))
        if np.abs(tilt_angle) > 90:
            print("Tilt angle has to be inside [-90, 90]")
        if s < 0 or s > self.mirror_size:
            print("s has to be inside [0, {}]".format(self.mirror_width))
        if t < 0 or t > self.mirror_size:
            print("t has to be inside [0, {}]".format(self.mirror_height))

    def get_x(self, nr_mirrors_x, nr_y, s, t):
        """
        Calculate the x coordinates depending on mirror properties
        """
        self.range_check(nr_mirrors_x, nr_y, self.tilt_angle_deg, s, t)
        cos = np.cos(self.tilt_angle_rad)
        matrix_product = s * (0.5 * (1 - cos) + cos) + t * (0.5 * (1 - cos))
        return matrix_product + (self.mirror_size + self.mirror_gap) * nr_mirrors_x - self.width / 2

    def get_y(self, nr_mirrors_x, nr_y, s, t):
        """
        Calculate the y coordinates depending on mirror properties
        """
        self.range_check(nr_mirrors_x, nr_y, self.tilt_angle_deg, s, t)
        cos = np.cos(self.tilt_angle_rad)
        matrix_product = s * (0.5 * (1 - cos)) + t * (0.5 * (1 - cos) + cos)
        return matrix_product + (self.mirror_size + self.mirror_gap) * nr_y - self.height / 2

    def get_z(self, nr_mirrors_x, nr_y, s, t):
        """
        Calculate the z coordinates depending on mirror properties
        """
        self.range_check(nr_mirrors_x, nr_y, self.tilt_angle_deg, s, t)
        sin = np.sin(self.tilt_angle_rad)
        matrix_product = 1 / np.sqrt(2) * (- s * sin + t * sin)
        return matrix_product
    
    def get_r0(self, nr_mirrors_x, nr_y):
        s0 = self.mirror_coords_x[0]
        t0 = self.mirror_coords_y[0]
        x0 = self.get_x(nr_mirrors_x, nr_y, s0, t0)
        y0 = self.get_y(nr_mirrors_x, nr_y, s0, t0)
        z0 = self.get_z(nr_mirrors_x, nr_y, s0, t0)
        return np.array([x0, y0, z0])
    
    def get_projection(self, nr_mirrors_x, nr_y, k_vector, r0):
        s0 = self.mirror_coords_x[0]
        s1 = self.mirror_coords_x[-1]
        t0 = self.mirror_coords_y[0]
        t1 = self.mirror_coords_y[-1]
        r1 = np.array([
            self.get_x(nr_mirrors_x, nr_y, s0, t1),
            self.get_y(nr_mirrors_x, nr_y, s0, t1),
            self.get_z(nr_mirrors_x, nr_y, s0, t1)])  # vector along y-axis (height) of mirror
        r2 = np.array([
            self.get_x(nr_mirrors_x, nr_y, s1, t0),
            self.get_y(nr_mirrors_x, nr_y, s1, t0),
            self.get_z(nr_mirrors_x, nr_y, s1, t0)])  # vector along x-axis (width) of mirror
        r_normal = np.cross(r1-r0, r2-r0)  # vector normal to the plane
        n = r_normal/np.linalg.norm(r_normal)  # norm to the plane
        return k_vector - np.dot(k_vector, n) * n
    
    def get_phase_shift(self, nr_mirrors_x, nr_y, s, t, k_proj, r0):
        r = np.array([self.get_x(nr_mirrors_x, nr_y, s, t), 
                      self.get_y(nr_mirrors_x, nr_y, s, t),
                      self.get_z(nr_mirrors_x, nr_y, s, t)])  # vector pointing towards point where phase should be calculated
        return np.dot(k_proj, r-r0)

    def get_surface_view(self, pixels_per_micron):
        st_per_mirror = pixels_per_micron * 10
        offset_x = self.width / 2
        offset_y = self.height / 2
        offset_z = 0
        width = int(self.nr_mirrors_x * (self.mirror_size + self.mirror_gap) * pixels_per_micron)
        height = int(self.nr_y * (self.mirror_size + self.mirror_gap) * pixels_per_micron)
        surface = np.zeros((width, height))

        for mx in range(self.nr_mirrors_x):
            for my in range(self.nr_y):
                for ss in range(st_per_mirror):
                    s = self.mirror_size / st_per_mirror * ss
                    for tt in range(st_per_mirror):
                        t = self.mirror_size / st_per_mirror * tt
                        y = self.get_y(mx, my, s, t)
                        z = self.get_z(mx, my, s, t)
                        x = self.get_x(mx, my, s, t)

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
    pass

if __name__ == "__main__":
    main()
