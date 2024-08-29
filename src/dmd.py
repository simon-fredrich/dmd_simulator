import numpy as np
import matplotlib.pyplot as plt
from metadata import MetaData
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
# init_notebook_mode(connected=True)


class Dmd2d:
    def __init__(self, tilt_angle_deg, m_size, m_gap, nr_m, nr_s) -> None:
        # dmd
        self.nr_m = nr_m
        self.nr_s = nr_s
        self.m_size = m_size
        self.m_gap = m_gap
        self.d_size = (m_size+m_gap)*nr_m-m_gap
        self.grid = self.create_grid()

        # mirror
        self.X, self.Z = self.create_mirror()

        # angles
        self.tilt_angle_rad = np.deg2rad(tilt_angle_deg)
        D2 = np.array([
            [np.cos(self.tilt_angle_rad), -np.sin(self.tilt_angle_rad)],
            [np.sin(self.tilt_angle_rad), np.cos(self.tilt_angle_rad)]
        ])
        self.x_rot, self.z_rot = np.dot(D2, np.vstack([self.X.flatten(), self.Z.flatten()]))

    def create_grid(self):
        grid = np.zeros(shape=(self.nr_m))
        x0 = (self.m_size-self.d_size)/2
        f = self.m_size+self.m_gap
        for mi in range(self.nr_m):
            pos = x0 + mi*f
            grid[mi] = pos
        return grid
    
    def create_mirror(self):
        x = np.linspace(-self.m_size/2, self.m_size/2, self.nr_s)
        z = np.zeros((self.nr_s))
        X, Z = np.meshgrid(x, z)
        return X, Z
    
    def plot(self, path="../out/default_2d.png"):
        surface_data = []
        for mi in range(self.nr_m):
            offset_x = self.grid[mi]
            X_rot = self.x_rot.reshape(self.X.shape) - offset_x
            Z_rot = self.z_rot.reshape(self.Z.shape)
            surface_data.append(go.Scatter(x=X_rot.flatten(), y=Z_rot.flatten(), showlegend=False))

        data = surface_data
        fig = go.Figure(data = data)
        fig.update_layout(
            width=800, height=800,
            yaxis=dict(range=[-self.d_size/4, self.d_size/4]),
            margin=dict(l=0, r=0, t=40, b=0))
        fig.write_image(path)
        iplot(fig)


class Dmd3d:
    def __init__(self, meta:MetaData) -> None:
        """
        Initialize a Dmd3d object from a MetaData object.

        Parameters
        ----------
        meta : MetaData
            A MetaData object containing the parameters for the DMD.

        Attributes
        ----------
        nr_m : int
            The number of mirrors in the x and y-direction.
        nr_s : int
            The number of mirror elements in the x and y-direction.
        m_size : float
            The size of each mirror element in the x and y-direction.
        m_gap : float
            The gap between each mirror element in the x and y-direction.
        d_size : float
            The total size of the DMD in the x and y-direction.
        grid : array
            An array containing the positions of each mirror element.
        pattern : array
            A 2D array containing the pattern to be projected onto the DMD.
        X, Y, Z : array
            2D arrays containing the positions of point sources within the initial mirror.
        tilt_angle_rad : float
            The tilt angle of the DMD in radians.
        rot_rad_z : float
            The rotation angle of the DMD in the z-direction in radians.
        """
        # dmd
        self.nr_m = meta.nr_m
        self.nr_s = meta.nr_s
        self.m_size = meta.m_size
        self.m_gap = meta.m_gap
        self.d_size = (meta.m_size+meta.m_gap)*meta.nr_m-meta.m_gap
        self.grid = self.create_grid()
        self.pattern = meta.pattern

        # mirror
        self.X, self.Y, self.Z = self.create_mirror()

        # angles
        self.tilt_angle_rad = np.deg2rad(meta.tilt_angle_deg)
        self.rot_rad_z = np.deg2rad(-45)

        # self.positions=self.compute_positions()
        
    def compute_position(self, mi, mj, tilt_state):
        """
        Compute the position of a single mirror in the DMD array.

        Parameters
        ----------
        mi : int
            x-index of the mirror in the array
        mj : int
            y-index of the mirror in the array
        tilt_state : int
            tilt state of the mirror (0 or 1)

        Returns
        -------
        position : np.ndarray
            (3, nr_s, nr_s) array of the position of the mirror
        """
        DY=self.rot_matrix_y(self.tilt_angle_rad*tilt_state)
        DZ=self.rot_matrix_z(self.rot_rad_z)
        D3=np.dot(DY, DZ)
        X, Y, Z=np.dot(D3, np.vstack([self.X.flatten(),
                                    self.Y.flatten(),
                                    self.Z.flatten()]))
        return np.array([X.reshape(self.X.shape)+self.grid[mi, mj][0],
                Y.reshape(self.Y.shape)+self.grid[mi, mj][1],
                Z.reshape(self.Z.shape)])
    
    def compute_positions(self):
        """
        Compute the positions of all mirrors in the DMD array.

        Returns
        -------
        positions : np.ndarray
            (nr_m, nr_m, 3, nr_s, nr_s) array of the positions of the mirrors
        """
        positions=np.zeros(shape=(self.nr_m, self.nr_m, 3, self.nr_s, self.nr_s))
        for mi in range(self.nr_m):
            for mj in range(self.nr_m):
                positions[mi, mj][0], positions[mi, mj][1], positions[mi, mj][2]=self.compute_position(mi, mj)
        return positions

    def create_grid(self):
        """
        Compute the positions of the grid points of the DMD array.

        The grid points are arranged in a hexagonal lattice, with the first
        point at the origin. The x and y coordinates of the grid points are
        computed as follows:

        x = sqrt(2)/2 * (m_size - d_size) + mi * sqrt(2)/2 * (m_size + m_gap)
        y = mj * sqrt(2)/2 * (m_size + m_gap)

        Returns
        -------
        grid : np.ndarray
            (nr_m, nr_m, 2) array of the positions of the grid points
        """
        grid = np.zeros(shape=(self.nr_m, self.nr_m, 2))
        x0 = np.array([np.sqrt(2)/2*(self.m_size-self.d_size), 0])
        f = np.sqrt(2)/2*(self.m_size+self.m_gap)
        row = f*np.array([1, -1])
        col = f*np.array([1, 1])
        for mi in range(self.nr_m):
            for mj in range(self.nr_m):
                pos = x0 + mi*row + mj*col
                grid[mi, mj] = pos
        return grid
    
    def create_mirror(self):
        """
        Compute the positions of the mirror surface points.

        The x and y coordinates are computed with np.linspace and the
        z coordinates are set to zero.

        Returns
        -------
        X, Y, Z : np.ndarray
            X, Y, Z coordinates of the mirror surface points
        """
        x = np.linspace(-self.m_size/2, self.m_size/2, self.nr_s)
        y = np.linspace(-self.m_size/2, self.m_size/2, self.nr_s)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        return X, Y, Z

    def rot_matrix_y(self, angle):
        """
        Compute the rotation matrix for a rotation around the y-axis by a given angle.

        Parameters
        ----------
        angle : float
            Angle in radians

        Returns
        -------
        rot_matrix_y : np.ndarray
            3x3 rotation matrix
        """
        
        rot_matrix_y = np.array(
            [[np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)]])
        return rot_matrix_y
        
    def rot_matrix_z(self, angle):
        """
        Compute the rotation matrix for a rotation around the z-axis by a given angle.

        Parameters
        ----------
        angle : float
            Angle in radians

        Returns
        -------
        rot_matrix_z : np.ndarray
            3x3 rotation matrix
        """
        rot_matrix_z = np.array(
            [[np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]])
        return rot_matrix_z

    def plot(self, path="../figures/default_3d.png"):
        surface_data = []
        scatter_data = []
        for mi in range(self.nr_m):
            for mj in range(self.nr_m):
                offset_x, offset_y = self.grid[mi, mj, 0], self.grid[mi, mj, 1]
                D3 = np.dot(self.rot_matrix_y(self.pattern[mi, mj]*self.tilt_angle_rad), self.rot_matrix_z(self.rot_rad_z))
                x_rot, y_rot, z_rot = np.dot(D3, np.vstack([self.X.flatten(), self.Y.flatten(), self.Z.flatten()]))
                X_rot = x_rot.reshape(self.X.shape) - offset_x
                Y_rot = y_rot.reshape(self.Y.shape) - offset_y
                Z_rot = z_rot.reshape(self.Z.shape)
                surface_data.append(go.Surface(x=X_rot, y=Y_rot, z=Z_rot, showscale=True if (mi==0 and mj==0) else False, colorbar=dict(title="Height") if (mi == 0 and mj == 0) else None))
                #scatter_data.append(go.Scatter3d(x=[X_rot[0, 0]], y=[Y_rot[0, 0]], z=[Z_rot[0, 0]], showlegend=False))

        data = surface_data + scatter_data
        fig = go.Figure(data = data)
        fig.update_layout(
            width=800, height=800, 
            scene=dict(
                zaxis=dict(
                    range=[
                        -self.d_size*np.sqrt(2)/2,
                        self.d_size*np.sqrt(2)/2])))
        fig.write_image(path)
        iplot(fig)

def main():
    dmd=Dmd3d(12, 10, 1, 10, 10, np.ones((10, 10)))
    dmd.create_mirror()

if __name__ == "__main__":
    main()
