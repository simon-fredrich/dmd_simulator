import os
import json
import numpy as np
import matplotlib.pyplot as plt
from metadata import MetaData  # Importiere deine MetaData-Klasse
from complex_field import ComplexField
from datetime import datetime

class SimulationSaver:
    def __init__(self, base_dir="./data/"):
        # Get the current time
        print(os.getcwd())
        current_time = datetime.now()
        self.formatted_current_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        self.base_dir = base_dir + self.formatted_current_time
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def save_plot(self, field:ComplexField, filename, plot_type='real', cmap='viridis'):
        """Speichert das gegebene Feld als Bilddatei."""
        if plot_type == 'real':
            field_to_plot = field.real()
            title = "Real Part"
        elif plot_type == 'imag':
            field_to_plot = field.real()
            title = "Imaginary Part"
        elif plot_type == 'abs':
            field_to_plot = field.abs()
            title = "Magnitude"
        elif plot_type == 'fft':
            fft, kx, ky = field.fft()
            title = "FFT Amplitude"
        else:
            raise ValueError("Invalid plot_type. Choose 'real', 'imag', or 'abs'.")

        plt.figure(figsize=(10, 8))
        plt.title(f"{title} of Field")
        if plot_type!="fft":
            plt.imshow(field_to_plot,
                    extent=(field.screen.x_min, field.screen.x_max,
                            field.screen.y_min, field.screen.y_max),
                    cmap=cmap)
        elif plot_type=="fft":
            plt.imshow(fft.abs(),
                    extent=(np.min(kx), np.max(kx),
                            np.min(ky), np.max(ky)),
                    cmap=cmap)
        plt.colorbar(label=title)
        plt.xlabel('X')
        plt.ylabel('Y')

        plot_type_filename = plot_type+"_"+filename
        filepath = os.path.join(self.base_dir, plot_type_filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Plot saved at {filepath}.")

    def save_pattern(self, pattern, filename):
        plt.imshow(pattern, cmap=plt.cm.gray)
        plot_type_filename="pattern_"+filename
        filepath=os.path.join(self.base_dir, plot_type_filename)
        plt.savefig(filepath)
        plt.close(f"Pattern saved at {filepath}.")

    def save_data(self, data, filename):
        """Speichert das gegebene Feld als NumPy-Datei."""
        filepath = os.path.join(self.base_dir, filename)
        np.save(filepath, data.mesh)
        print(f"Data saved at {filepath}")

    def save_metadata(self, meta: MetaData, filename):
        """Speichert eine Instanz der MetaData-Klasse als JSON-Datei."""
        metadata_dict = {
            'created': self.formatted_current_time,
            'tilt_angle_deg': meta.tilt_angle_deg,
            'm_size': meta.m_size,
            'm_gap': meta.m_gap,
            'nr_m': f"{meta.nr_m}x{meta.nr_m}",
            'nr_s': f"{meta.nr_s}x{meta.nr_s}",
            'pattern': meta.pattern.shape,
            'wavelength': meta.wavelength,
            'pixels:': f"{meta.pixels}x{meta.pixels}",
            'incident_angle_deg': meta.incident_angle_deg,
            'Computing Time (seconds)': meta.computing_time,
            'Computing Time': f"{meta.computing_time//60} min {meta.computing_time%60} sec"
        }
        
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metadata_dict, f, indent=4)
        print(f"Metadata saved at {filepath}")

    def save_full_simulation(self, field, meta: MetaData, base_filename):
        """Speichert das Feld, Plots und Metadaten in einem Schritt."""
        plot_filename = f"{base_filename}_plot.png"
        data_filename = f"{base_filename}_data.npy"
        pattern_filename=f"{base_filename}_pattern.png"
        metadata_filename = f"{base_filename}_meta.json"

        # Speichere Plot
        self.save_plot(field, plot_filename, plot_type='abs')
        self.save_plot(field, plot_filename, plot_type='real')
        self.save_plot(field, plot_filename, plot_type='imag')
        self.save_plot(field, plot_filename, plot_type='fft')
        self.save_pattern(meta.pattern, pattern_filename)
        
        # Speichere Rohdaten
        self.save_data(field, data_filename)
        
        # Speichere Metadaten (nutzt jetzt die MetaData-Klasse)
        self.save_metadata(meta, metadata_filename)

        print(f"Full simulation saved with base filename: {base_filename}")
