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
        self.rotated_points = self.do_rotation(xspan, yspan, rot_rad_y=self.tilt_angle_rad)
        self.coords = np.zeros((nr_mirrors, nr_mirrors, 3, nr_sources, nr_sources))

        # quantities
        self.nr_mirrors = nr_mirrors
        self.nr_sources = nr_sources
        self.nr_sources_total = np.square(nr_mirrors * nr_sources)
        self.size = (mirror_size + mirror_gap) * nr_mirrors - mirror_gap
        self.size_diag = np.sqrt(2)*self.size

    def do_rotation(self, xspan, yspan, rot_rad_x=0, rot_rad_y=0, rot_rad_z=np.deg2rad(45+90)):
        """Generate a 3D meshgrid and rotate it by rot_rad_x, rot_rad_y, and rot_rad_z radians."""

        # Rotation matrices for x, y, and z axes
        rot_matrix_x = np.array([[1, 0, 0],
                            [0, np.cos(rot_rad_x), -np.sin(rot_rad_x)],
                            [0, np.sin(rot_rad_x), np.cos(rot_rad_x)]])
        
        rot_matrix_y = np.array([[np.cos(rot_rad_y), 0, np.sin(rot_rad_y)],
                            [0, 1, 0],
                            [-np.sin(rot_rad_y), 0, np.cos(rot_rad_y)]])
        
        rot_matrix_z = np.array([[np.cos(rot_rad_z), -np.sin(rot_rad_z), 0],
                            [np.sin(rot_rad_z), np.cos(rot_rad_z), 0],
                            [0, 0, 1]])

        D3 = np.dot(rot_matrix_y, rot_matrix_z)

        # Create a 3D meshgrid
        X, Y = np.meshgrid(xspan, yspan)
        Z = np.zeros_like(X)

        # Apply rotation to the grid points
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()

        points = np.vstack((X_flat, Y_flat, Z_flat))
        rotated_points = np.dot(D3, points)

        X_rotated = rotated_points[0, :].reshape(X.shape)
        Y_rotated = rotated_points[1, :].reshape(Y.shape)
        Z_rotated = rotated_points[2, :].reshape(Z.shape)

        return X_rotated, Y_rotated, Z_rotated
    
    def get_x(self, mx, my):
        return self.rotated_points[0] + np.sqrt(2)/2*(self.mirror_size+self.mirror_gap)*(mx - my)
    
    def get_y(self, mx, my):
        return self.rotated_points[1] + np.sqrt(2)/2*(self.mirror_size+self.mirror_gap)*(mx + my + 1) - self.size_diag/2
    
    def get_z(self, mx, my):
        return self.rotated_points[2]
    
    def get_coords(self, mx, my):
        # Extract the rotated coordinates and arrange them onto dmd plane
        x = self.get_x(mx, my)
        y = self.get_y(mx, my)
        z = self.get_z(mx, my)

        # save position for later use
        self.coords[mx, my, 0] = x
        self.coords[mx, my, 1] = y
        self.coords[mx, my, 2] = z

        return x, y, z
    
    def plot(self, show_info=False):
        from plotly.offline import init_notebook_mode, iplot
        import plotly.graph_objs as go
        init_notebook_mode(connected=True)

        fig = go.Figure()
        for mx in range(self.nr_mirrors):
            for my in range(self.nr_mirrors):
                x = self.get_x(mx, my)
                y = self.get_y(mx, my)
                z = self.get_z(mx, my)

                x0 = x.flatten()[0]
                y0 = y.flatten()[0]
                z0 = z.flatten()[0]

                mirror = go.Surface(x=x, y=y, z=z, showscale=(True if (mx == 0 and my == 0) else False))
                fig.add_trace(mirror)
                if show_info:
                    fig.add_trace(
                        go.Scatter3d(x=[x0],
                                    y=[y0],
                                    z=[z0],
                                    mode='markers+text',
                                    marker=dict(
                                            size=2,
                                            colorscale='Viridis',   # choose a colorscale
                                            opacity=0.8),
                                    text=f"{mx}, {my}",
                                    showlegend=False
                        )
                    )
        
        # Set y-axis limits
        fig.update_layout(
            scene=dict(
                zaxis=dict(
                    range=[-self.size, self.size]  # Set the z-axis range
                )
            ),
            title='DMD Surface view', autosize=False,
            width=500, height=500,
            margin=dict(l=65, r=50, b=65, t=90)
        )
        iplot(fig)

def main():
    pass

if __name__ == "__main__":
    main()
