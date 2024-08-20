import os
import json
import numpy as np
import matplotlib.pyplot as plt
from metadata import MetaData  # Importiere deine MetaData-Klasse

class SimulationSaver:
    def __init__(self, base_dir="./results"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def save_plot(self, field, x_range, y_range, filename, plot_type='real', cmap='viridis'):
        """Speichert das gegebene Feld als Bilddatei."""
        if plot_type == 'real':
            field_to_plot = np.real(field)
            title = "Real Part"
        elif plot_type == 'imag':
            field_to_plot = np.imag(field)
            title = "Imaginary Part"
        elif plot_type == 'abs':
            field_to_plot = np.abs(field)
            title = "Magnitude"
        else:
            raise ValueError("Invalid plot_type. Choose 'real', 'imag', or 'abs'.")

        plt.figure(figsize=(10, 8))
        plt.title(f"{title} of Field")
        plt.imshow(field_to_plot, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), cmap=cmap)
        plt.colorbar(label=title)
        plt.xlabel('X')
        plt.ylabel('Y')

        filepath = os.path.join(self.base_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Plot saved at {filepath}")

    def save_data(self, data, filename):
        """Speichert das gegebene Feld als NumPy-Datei."""
        filepath = os.path.join(self.base_dir, filename)
        np.save(filepath, data)
        print(f"Data saved at {filepath}")

    def save_metadata(self, meta: MetaData, filename):
        """Speichert eine Instanz der MetaData-Klasse als JSON-Datei."""
        metadata_dict = {
            'tilt_angle_deg': meta.tilt_angle_deg,
            'm_size': meta.m_size,
            'm_gap': meta.m_gap,
            'nr_m': meta.nr_m,
            'nr_s': meta.nr_s,
            'pattern': meta.pattern.tolist(),  # Konvertiere NumPy-Array in eine Liste
            'wavelength': meta.wavelength,
            'incident_angle_deg': meta.incident_angle_deg
        }
        
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metadata_dict, f, indent=4)
        print(f"Metadata saved at {filepath}")

    def save_full_simulation(self, field, x_range, y_range, meta: MetaData, base_filename):
        """Speichert das Feld, Plots und Metadaten in einem Schritt."""
        plot_filename = f"{base_filename}_plot.png"
        data_filename = f"{base_filename}_data.npy"
        metadata_filename = f"{base_filename}_meta.json"

        # Speichere Plot
        self.save_plot(field, x_range, y_range, plot_filename, plot_type='abs')
        
        # Speichere Rohdaten
        self.save_data(field, data_filename)
        
        # Speichere Metadaten (nutzt jetzt die MetaData-Klasse)
        self.save_metadata(meta, metadata_filename)

        print(f"Full simulation saved with base filename: {base_filename}")
