import numpy as np
import matplotlib.pyplot as plt

class Mirror1D:
    def __init__(self, size, tilt_angle):
        self.size = size
        self.tilt_angle = tilt_angle * np.pi / 180
        self.middle_x = self.size / 2.0
        self.middle_y = 0.0

    def get_x(self, x):
        y = 0
        cos = np.cos(self.tilt_angle)
        sin = np.sin(self.tilt_angle)
        target_x = (x - self.middle_x) * cos - (y - self.middle_y) * sin + self.middle_x
        return target_x

    def get_y(self, x):
        y = 0
        cos = np.cos(self.tilt_angle)
        sin = np.sin(self.tilt_angle)
        target_y = (x - self.middle_x) * sin + (y - self.middle_y) * cos + self.middle_y
        return target_y

    def k_parallel(self, k_vector):
        b = np.array([self.get_x(self.size) - self.get_x(0), self.get_y(self.size) - self.get_y(0)])
        b_norm = b / np.linalg.norm(b)
        result = np.dot(k_vector, b_norm) * b_norm
        return result

    def phase_at(self, x, k_vector):
        r = np.array([self.get_x(x) - self.get_x(0), self.get_y(x) - self.get_y(0)])
        result = np.dot(self.k_parallel(k_vector), r)%(2*np.pi)
        return result