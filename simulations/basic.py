import numpy as np
import sys
sys.path.insert(1, './src/')
from metadata import MetaData
from dmd import Dmd3d
from simulation import Simulation3d
from simulation_saver import SimulationSaver
import time
from datetime import datetime
# Get the current time
current_time = datetime.now()
formatted_current_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")

# Erstelle eine Instanz der MetaData-Klasse
meta=MetaData()
meta.nr_s=60
meta.nr_m=30
meta.pixels=512
meta.incident_angle_deg=70
meta.update_pattern()

dmd3d_instance = Dmd3d(meta)

# Starte die Simulation
sim3d = Simulation3d(dmd3d_instance, meta)

lim=dmd3d_instance.d_size*np.sqrt(2)
total_field = sim3d.compute_field(pixels=meta.pixels, 
                                  x_min=-lim, x_max=lim, y_min=-lim, y_max=lim, z=10)

# Initialisiere den Saver
saver = SimulationSaver(base_dir="./data/"+formatted_current_time)

# Speichere das Ergebnis der Simulation, einschließlich der MetaData-Instanz
saver.save_full_simulation(total_field, sim3d.dmd.grid[:, :, 0], sim3d.dmd.grid[:, :, 1], meta, "basic_sim3d")
