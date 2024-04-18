import numpy as np
import matplotlib.pyplot as plt

class Mirror:
    def __init__(self, support_vector, width, height, tilt_angle) -> None:
        if (width != height):
            print("Mirror is not square")
        self.support_vector = support_vector
        self.width = width
        self.height = height
        self.tilt_angle = tilt_angle # in degrees

    def set_tilt(self, tilt_angle):
        self.tilt_angle = tilt_angle

    


def main():
    my_mirror = Mirror(0, 0, 10, 10, 12)


if __name__ == "__main__":
    main()