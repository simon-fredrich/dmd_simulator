import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)


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
    def __init__(self, tilt_angle_deg, m_size, m_gap, nr_m, nr_s, pattern) -> None:
        # dmd
        self.nr_m = nr_m
        self.nr_s = nr_s
        self.m_size = m_size
        self.m_gap = m_gap
        self.d_size = (m_size+m_gap)*nr_m-m_gap
        self.grid = self.create_grid()
        self.pattern = pattern

        # mirror
        self.X, self.Y, self.Z = self.create_mirror()

        # angles
        self.tilt_angle_rad = np.deg2rad(tilt_angle_deg)
        self.rot_rad_z = np.deg2rad(-45)
        
    
    def create_grid(self):
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
        x = np.linspace(-self.m_size/2, self.m_size/2, self.nr_s)
        y = np.linspace(-self.m_size/2, self.m_size/2, self.nr_s)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        return X, Y, Z

    def rot_matrix_y(self, angle):
        rot_matrix_y = np.array(
            [[np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)]])
        return rot_matrix_y
        
    def rot_matrix_z(self, angle):
        rot_matrix_z = np.array(
            [[np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]])
        return rot_matrix_z

    def plot(self, path="../out/default_3d.png"):
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
    pass

if __name__ == "__main__":
    main()
