import numpy as np
import matplotlib.pyplot as plt
from mirror_1d import Mirror1D


mirror_size = 10
alpha = 12
mirror = Mirror1D(mirror_size, alpha)
wavelength = 1
phi = -30 * np.pi / 180
k = 2 * np.pi / wavelength
k_vector = k * np.array([np.cos(phi), np.sin(phi)])

print(mirror.phase_at(1, k_vector))

s = np.linspace(0, mirror.size)

field = np.zeros((10), dtype=complex)

field_length = 10
field_x = 0
field_y = 4
for i in range(len(field)):
    sum = 0 + 0j
    for x in s:
        r_0 = np.array([field_x - mirror.get_x(x), field_y - mirror.get_y(0)])
        k_new = k*(r_0/np.linalg.norm(r_0))
        phase = mirror.phase_at(mirror.get_x(x), k_vector)
        sum += np.exp(1j*(np.dot(k_new, r_0) + phase))
    field[i] = sum

print(field)
    
def visualize_mirror():
    for x in s:
        plt.scatter(x, 0, color="blue")
        plt.scatter(mirror.get_x(x), mirror.get_y(x), color="red")

    plt.scatter(mirror.k_parallel(k_vector)[0], mirror.k_parallel(k_vector)[1], color="green")
    plt.scatter(1.5*mirror.k_parallel(k_vector)[0], 1.5*mirror.k_parallel(k_vector)[1], color="green")
    plt.scatter(2*mirror.k_parallel(k_vector)[0], 2*mirror.k_parallel(k_vector)[1], color="green")


def visualize_field(field):
    plt.plot(range(len(field)), np.linalg.norm(field))
    plt.xlabel("x")
    plt.ylabel("intensity")

visualize_field(field)
# visualize_mirror()

plt.xlim(0, 10)
plt.ylim(-5, 5)
plt.show()