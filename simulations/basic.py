import sys
sys.path.insert(1, '../src/')
from metadata import MetaData
from dmd import Dmd3d
from simulation import Simulation3d
from simulation_saver import SimulationSaver

# Erstelle eine Instanz der MetaData-Klasse
meta_instance = MetaData()
dmd3d_instance = Dmd3d(meta_instance)

# Starte die Simulation
sim3d = Simulation3d(dmd3d_instance, meta_instance)
total_field = sim3d.compute_field(pixels=512, x_min=-100, x_max=100, y_min=-100, y_max=100, z=10)

# Initialisiere den Saver
saver = SimulationSaver(base_dir="../data/basic_results")

# Speichere das Ergebnis der Simulation, einschließlich der MetaData-Instanz
saver.save_full_simulation(total_field.mesh, sim3d.dmd.grid[:, :, 0], sim3d.dmd.grid[:, :, 1], meta_instance, "basic_sim3d")
