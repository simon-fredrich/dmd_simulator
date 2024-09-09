import numpy as np

class MetaData:
    def __init__(self) -> None:
        # default dmd parameters
        self.tilt_angle_deg=12
        self.m_size=10
        self.m_gap=1
        self.nr_m=11
        self.nr_s=51
        self.pattern=np.ones((self.nr_m, self.nr_m))
        self.wavelength=0.5
        self.pixels=128

        # default simulation parameters
        self.incident_angle_deg=-24
        
        self.computing_time=None

    def set_nr_s(self, nr_s):
        self.nr_s=nr_s

    def set_nr_m(self, nr_m):
        self.nr_m=nr_m
        self.update_pattern()

    def set_pixels(self, pixels):
        self.pixels=pixels

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