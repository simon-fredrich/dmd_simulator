import numpy as np
import matplotlib.pyplot as plt


class Image:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.values = np.zeros(shape=(width, height))

    def set_value(self, x, y, value):
        self.values[y, x] = value

    def save(self, path):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.values, cmap='viridis', origin='lower')
        plt.colorbar(label='Z Value')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Surface Visualization')
        plt.savefig(path, format='pdf')


def main():
    image = Image(100, 100)
    image.set_value(3, 5, 1)
    image.save("../out/test_image.pdf")


if __name__ == "__main__":
    main()
