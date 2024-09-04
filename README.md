# DMD Simulation Project

This project simulates the behavior of **Digital Micromirror Devices (DMD)** and models the interaction between light waves and the DMD surface using both 2D and 3D simulations.

## Table of Contents
1. [Overview](#overview)
2. [Development Environment](#development-environment)
3. [Libraries Used](#libraries-used)
4. [Class Structure](#class-structure)
5. [Key Classes and Methods](#key-classes-and-methods)
6. [Simulation Parameters](#simulation-parameters)
7. [How to Run the Project](#how-to-run-the-project)
8. [License](#license)

## Overview
This project was developed to simulate the optical behavior of DMDs using Python. It includes:
- A 2D simulation to develop a basic understanding of light interaction with DMD mirrors.
- A 3D simulation for a more realistic model, where mirrors are rotated and tilted to simulate real-world scenarios.

## Development Environment
- **IDE**: [Visual Studio Code (VSCode)](https://code.visualstudio.com/)
- **Interactive Development**: Jupyter Notebooks embedded within VSCode were used for iterative development and debugging.
- **Version Control**: Git was used for managing source control.

## Libraries Used
The following Python libraries are key to the project's development:
- **NumPy**: For numerical operations and efficient handling of N-dimensional arrays.
- **Matplotlib**: For visualizing the simulation results in 2D and 3D plots.
- **Plotly**: Used during the development process for interactive visualizations.
  
## Class Structure
The project uses a well-structured object-oriented approach, encapsulating the simulation logic into the following classes:
- **`Dmd2d`**: Simulates the behavior of a DMD in two dimensions.
- **`Dmd3d`**: Extends the simulation to three dimensions, accounting for rotation and tilt.
- **`MetaData`**: Manages and stores simulation parameters such as mirror size, tilt angles, light wavelength, etc.

### Dmd2d Class
- Simulates DMD behavior on a 2D plane.
- Parameters include the number of mirrors (`nr_m`), mirror size (`m_size`), and gaps between mirrors (`m_gap`).

### Dmd3d Class
- Extends the 2D simulation to three dimensions, considering the mirror's orientation and tilt.
- Simulates light interactions in 3D space.

### MetaData Class
- Stores simulation parameters (e.g., tilt angles, mirror sizes, wavelength).
- Allows easy adjustment and storage of simulation settings.

## Key Classes and Methods

### MetaData Class
- **Attributes**:
  - `tilt_angle_deg`: Tilt angle of mirrors in degrees.
  - `m_size`: Size of individual mirrors.
  - `m_gap`: Distance between adjacent mirrors.
  - `nr_m`: Number of mirrors along each dimension.
  - `pattern`: Defines the hologram pattern applied to the DMD.
  
- **Methods**:
  - `update_pattern()`: Updates the pattern projected onto the DMD.
  - `output()`: Outputs all parameters in a structured format.

### Dmd2d Class
- **Method**: `create_grid()`
  - Creates a 2D grid representing the positions of mirrors on the DMD surface.
  
### Dmd3d Class
- **Method**: `create_grid()`
  - Creates a 3D grid and accounts for the mirror's tilt and rotation in space.

## Simulation Parameters
The following parameters are crucial for configuring the simulation:

| Parameter       | Description                                               |
|-----------------|-----------------------------------------------------------|
| `tilt_angle_deg`| Mirror tilt angle in degrees                               |
| `m_size`        | Size of each mirror in the DMD                             |
| `m_gap`         | Gap between adjacent mirrors                               |
| `nr_m`          | Number of mirrors in each dimension                        |
| `wavelength`    | Wavelength of the incoming wave                            |
| `pattern`       | Pattern or hologram applied to the DMD surface             |

## How to Run the Project
1. **Install the required dependencies**:
   ```bash
   pip install numpy matplotlib plotly
   ```

2. **Run Jupyter Notebooks** to test and visualize simulations.
   - Use VSCode or another Jupyter Notebook interface.

3. **Run the simulation** by initializing the `Dmd2d` or `Dmd3d` class, and configure the simulation parameters using the `MetaData` class.

## License
This project is provided under the MIT License.