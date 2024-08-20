import numpy as np

class MetaData:
    def __init__(self) -> None:
        # default dmd parameters
        self.tilt_angle_deg=0
        self.m_size=10
        self.m_gap=1
        self.nr_m=10
        self.nr_s=50
        self.pattern=np.ones((self.nr_m, self.nr_m))
        self.wavelength=0.5

        # default simulation parameters
        self.incident_angle_deg=0

    def update_pattern(self):
        self.pattern=np.ones((self.nr_m, self.nr_m))