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
    def __init__(self, tilt_angle, mirror_size, mirror_gap, nr_mirrors, nr_sources) -> None:
        self.tilt_angle_deg = tilt_angle
        self.tilt_angle_rad = np.deg2rad(tilt_angle)
        self.mirror_size = mirror_size
        self.mirror_gap = mirror_gap

        # construct tilted mirror
        xspan = np.linspace(-mirror_size/2, mirror_size/2, nr_sources)
        yspan = np.linspace(-mirror_size/2, mirror_size/2, nr_sources)
        zspan = np.zeros_like(xspan)
        self.rotated_mesh = self.DoRotation3D(xspan, yspan, zspan, RotRadY=self.tilt_angle_rad, RotRadZ=np.deg2rad(45))
        self.coords = np.zeros((nr_mirrors, nr_mirrors, 3, nr_sources, nr_sources, nr_sources))

        # quantities
        self.nr_mirrors = nr_mirrors
        self.nr_sources = nr_sources
        self.nr_sources_total = np.square(nr_mirrors * nr_sources)
        self.size = (mirror_size + mirror_gap) * nr_mirrors - mirror_gap
        self.size_diag = np.sqrt(2)*self.size

    def DoRotation3D(self, xspan, yspan, zspan, RotRadX=0, RotRadY=0, RotRadZ=0):
        """Generate a 3D meshgrid and rotate it by RotRadX, RotRadY, and RotRadZ radians."""

        # Rotation matrices for x, y, and z axes
        RotMatrixX = np.array([[1, 0, 0],
                            [0, np.cos(RotRadX), -np.sin(RotRadX)],
                            [0, np.sin(RotRadX), np.cos(RotRadX)]])
        
        RotMatrixY = np.array([[np.cos(RotRadY), 0, np.sin(RotRadY)],
                            [0, 1, 0],
                            [-np.sin(RotRadY), 0, np.cos(RotRadY)]])
        
        RotMatrixZ = np.array([[np.cos(RotRadZ), -np.sin(RotRadZ), 0],
                            [np.sin(RotRadZ), np.cos(RotRadZ), 0],
                            [0, 0, 1]])

        # Combined rotation matrix
        RotMatrix = RotMatrixY @ RotMatrixZ

        # Create a 3D meshgrid
        x, y, z = np.meshgrid(xspan, yspan, zspan, indexing='ij')

        # Stack the meshgrid arrays into a single 4D array
        grid = np.stack([x, y, z], axis=-1)

        # Apply the rotation matrix
        rotated_grid = np.einsum('ij,klmj->klmi', RotMatrix, grid)

        return rotated_grid
    
    def get_coords(self, mx, my):
        # Extract the rotated coordinates and arrange them onto dmd plane
        x_rot = self.rotated_mesh[..., 0] + np.sqrt(2)/2*(self.mirror_size+self.mirror_gap)*(mx - my)
        y_rot = self.rotated_mesh[..., 1] + np.sqrt(2)/2*(self.mirror_size+self.mirror_gap)*(mx + my + 1) - self.size_diag/2
        z_rot = self.rotated_mesh[..., 2]

        # save position for later use
        self.coords[mx, my, 0] = x_rot
        self.coords[mx, my, 1] = y_rot
        self.coords[mx, my, 2] = z_rot

        return x_rot, y_rot, z_rot

def main():
    pass

if __name__ == "__main__":
    main()
