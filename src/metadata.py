import numpy as np

class MetaData:
    def __init__(self, tilt_angle_deg, incident_angle_deg, m_size, m_gap, nr_m, nr_s, wavelength, pixels) -> None:
        self.tilt_angle_deg=tilt_angle_deg
        self.incident_angle_deg=incident_angle_deg
        self.m_size=m_size
        self.m_gap=m_gap
        self.nr_m=nr_m
        self.nr_s=nr_s
        self.wavelength=wavelength
        self.pixels=pixels

    def set_nr_s(self, nr_s):
        self.nr_s=nr_s

    def set_nr_m(self, nr_m):
        self.nr_m=nr_m
        self.update_pattern()

    def set_pixels(self, pixels):
        self.pixels=pixels

    def set_m_size(self, m_size):
        self.m_size=m_size

    def set_m_gap(self, m_gap):
        self.m_gap=m_gap

    def set_tilt_angle_deg(self, tilt_angle_deg):
        self.tilt_angle_deg=tilt_angle_deg

    def set_wavelength(self, wavelength):
        self.wavelength=wavelength

    def update_pattern(self):
        self.pattern=np.ones((self.nr_m, self.nr_m))

    def output(self):
        # Create an output string with the settings
        output_str = (
            f"\nTilt Angle (deg): {self.tilt_angle_deg}\n"
            f"Mirror Size: {self.m_size}\n"
            f"Mirror Gap: {self.m_gap}\n"
            f"Number of Mirrors: {self.nr_m}x{self.nr_m}\n"
            f"Number of Sources: {self.nr_s}x{self.nr_s}\n"
            f"Pattern Shape: {self.pattern.shape}\n"
            f"Wavelength: {self.wavelength}\n"
            f"Pixels: {self.pixels}x{self.pixels}\n"
            f"Incident Angle (deg): {self.incident_angle_deg}\n"
        )
        return output_str