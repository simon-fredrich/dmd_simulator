# DMD Simulation Project

This project simulates the behavior of **Digital Micromirror Devices (DMD)** and models the interaction between light waves and the DMD surface using both 2D and 3D simulations. The simulations are designed to help understand light interference, diffraction, and optical field behavior as it interacts with DMDs.

## Table of Contents
1. [Overview](#overview)
2. [Development Environment](#development-environment)
3. [Libraries Used](#libraries-used)
4. [Class Structure](#class-structure)
5. [Key Classes and Methods](#key-classes-and-methods)
6. [Simulation Parameters](#simulation-parameters)
7. [How to Run the Project](#how-to-run-the-project)
8. [Simulation of Optical Fields](#simulation-of-optical-fields)
9. [Fourier Transformation](#fourier-transformation)
10. [License](#license)

## Overview
The goal of this project is to simulate the optical behavior of **Digital Micromirror Devices (DMD)**, which are essential components in many modern optical systems. The simulation covers both **2D** and **3D** models of DMDs to analyze the light distribution, diffraction, and interference patterns. The models are highly flexible, allowing users to simulate various optical setups.

## Development Environment
The project was developed in **Visual Studio Code (VSCode)**, leveraging the **Jupyter Notebooks** extension for interactive development and debugging.

- **IDE**: Visual Studio Code (VSCode)
- **Interactive Development**: Jupyter Notebooks
- **Version Control**: Git

## Libraries Used

### Core Libraries:
- **NumPy**: Used for numerical calculations and manipulation of N-dimensional arrays. Essential for handling large datasets efficiently and performing mathematical operations on optical fields.
- **Matplotlib**: Provides 2D and 3D plotting functionalities to visualize the simulation results.
- **Plotly**: Used for interactive plots during the development phase to help analyze visual data more effectively.

## Class Structure

### Main Classes:
1. **`MetaData`**: Handles the configuration and storage of simulation parameters such as mirror size, wavelength, and mirror tilt angles.
2. **`Dmd2d`**: Simulates DMD behavior in two dimensions, allowing for basic light interaction studies.
3. **`Dmd3d`**: Extends the 2D DMD simulation to three dimensions, adding realistic tilt and rotation effects on mirrors.
   
### Additional Classes:
- **`ComplexField`**: Handles complex optical fields, including operations such as Fourier transformations and field shifting.
- **`Simulation2d`** and **`Simulation3d`**: These classes model the full simulation behavior, providing methods for calculating incident and reflected fields.

## Key Classes and Methods

### MetaData Class
The `MetaData` class stores and manages all important simulation parameters:
- **Attributes**:
  - `tilt_angle_deg`: Mirror tilt angle (in degrees).
  - `m_size`: The size of each mirror.
  - `m_gap`: Distance between mirrors.
  - `nr_m`: Number of mirrors along each dimension.
  - `wavelength`: Wavelength of the light used in the simulation.
  - `pattern`: The hologram pattern applied to the DMD surface.

- **Methods**:
  - `update_pattern()`: Updates the current DMD pattern, setting each mirror to either "on" or "off".
  - `output()`: Outputs the current simulation parameters for reference.

### Dmd2d Class
The `Dmd2d` class models the mirrors as a 2D array on a plane. It calculates the optical effects based on mirror size, tilt, and incoming light angles.

- **Key Methods**:
  - `create_grid()`: Creates a grid of mirrors based on their physical properties and positions them in a 2D space.
  - `create_mirror()`: Defines the surface area of each mirror based on the grid configuration.

### Dmd3d Class
The `Dmd3d` class extends the `Dmd2d` model into 3D, allowing the mirrors to rotate and tilt in space. It captures more complex light interactions due to this added dimensionality.

- **Key Methods**:
  - `create_grid()`: Places mirrors in a 3D grid while accounting for rotation and tilt.
  - `create_mirror()`: Calculates the position of each mirror in 3D space and its effect on light reflection.

### ComplexField Class
This class handles the simulation of complex optical fields, including:
- **shift()**: Shifts the optical field based on mirror movement.
- **fft()**: Performs a Fourier transformation on the field to analyze diffraction and interference patterns.

## Simulation Parameters

These parameters control the behavior of the DMD simulation. You can adjust these parameters via the `MetaData` class.

| Parameter       | Description                                               |
|-----------------|-----------------------------------------------------------|
| `tilt_angle_deg`| Mirror tilt angle in degrees                               |
| `m_size`        | Size of each mirror in the DMD                             |
| `m_gap`         | Gap between adjacent mirrors                               |
| `nr_m`          | Number of mirrors in each dimension                        |
| `wavelength`    | Wavelength of the incident light                           |
| `pattern`       | Hologram pattern applied to the DMD surface                |
| `computing_time`| Time taken for the optical field simulation                |

## How to Run the Project

### Installation:
1. Install the required dependencies:
   ```bash
   pip install numpy matplotlib plotly
   ```

2. Run the simulations in **Jupyter Notebooks** or directly in Python using VSCode.

### Example Usage:
- Initialize the DMD object (either `Dmd2d` or `Dmd3d`).
- Set simulation parameters through the `MetaData` class.
- Use the provided methods to visualize and analyze the optical fields.

```python
from dmd_simulation import Dmd2d, MetaData

metadata = MetaData(tilt_angle_deg=12, m_size=10, wavelength=0.5)
dmd2d = Dmd2d(metadata)
dmd2d.create_grid()
dmd2d.create_mirror()
```

## Simulation of Optical Fields

The simulation of optical fields involves calculating both the **incident** and **reflected fields** as light interacts with the DMD mirrors. This is handled by the `Simulation2d` and `Simulation3d` classes.

### Key Steps:
1. **Initialize Parameters**: Define the simulation parameters such as mirror size, tilt angles, and wavelength.
2. **Compute Incident Field**: Calculate the field of the incoming light wave based on its angle and wavelength.
3. **Reflect Field**: Calculate the reflected field based on the mirror angles and positions, considering interference and diffraction effects.

## Fourier Transformation

The Fourier transformation is used to analyze the spatial frequency of the optical fields. This allows for detailed analysis of diffraction and interference patterns produced by the DMD.

- **Key Method**:
  - `fft()`: Performs a 2D Fourier transform on the optical fields to study the resulting diffraction patterns.

## License
This project is licensed under the MIT License.