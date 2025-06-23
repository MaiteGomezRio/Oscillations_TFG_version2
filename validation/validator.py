# -*- coding: utf-8 -*-
#validation/validator.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def export_xic_signals_combined_csv(rts, xic_signals, modulated_signals, residual_signals, output_csv_path):
    """
    Exporta un único CSV con los XIC, señales moduladas y residuales para cada m/z.
    """
    df = pd.DataFrame({'RT': rts})

    for target_mz in xic_signals:
        mz_str = f"{target_mz:.4f}"
        df[f"XIC_{mz_str}"] = xic_signals[target_mz]
        df[f"Modulated_{mz_str}"] = modulated_signals[target_mz]
        df[f"Residual_{mz_str}"] = residual_signals[target_mz]

    # Aplicar formato español a todos los valores numéricos
    df_formatted = df.map(
        lambda x: f"{x:,.6f}".replace(",", "X").replace(".", ",").replace("X", ".")
        if isinstance(x, (float, int)) else x
    )

    # Guardar con separador de columnas ';' (opcional pero común en CSVs en español)
    df_formatted.to_csv(output_csv_path, index=False, sep=';')
    print(f"Exportado CSV combinado a: {output_csv_path}")
    
def plot_ms_experiment_3d(ms_experiment):
    """
    Grafica un experimento MS (MSExperiment) en 3D con RT, m/z, y intensidad, con color según la intensidad.
    """
    rts = []
    mzs = []
    intensities = []

    for spec in ms_experiment:
        rt = spec.getRT()
        mz, inten = spec.get_peaks()

        if len(mz) == 0:
            continue  # ignorar espectros vacíos

        rts.extend([rt] * len(mz))
        mzs.extend(mz)
        intensities.extend(inten)

    rts = np.array(rts)
    mzs = np.array(mzs)
    intensities = np.array(intensities)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Crear el scatter plot con mapeo de color
    sc = ax.scatter(rts, mzs, intensities, c=intensities, cmap='viridis', marker='o', s=5, alpha=0.8)

    # Añadir colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("Intensity")

    ax.set_xlabel("Retention Time (s)")
    ax.set_ylabel("m/z")
    ax.set_zlabel("Intensity")
    ax.set_title("3D MS Corrected Map (Color = Intensity)")

    plt.tight_layout()
    plt.show()

def plot_xic_from_map(ms_map, target_mz, mz_tol=0.01):
    rts = []
    intensities = []
    for spec in ms_map:
        if spec.getMSLevel() != 1:
            continue
        mzs, intens = spec.get_peaks()
        rt = spec.getRT()
        for mz, intensity in zip(mzs, intens):
            if abs(mz - target_mz) <= mz_tol:
                rts.append(rt)
                intensities.append(intensity)
                break
    plt.plot(rts, intensities)
    plt.xlabel("Retention Time (s)")
    plt.ylabel(f"Intensity at {target_mz} m/z")
    plt.title(f"XIC of {target_mz}")
    plt.show()

def plot_all(rts, target_mz, xic, modulated_signal, residual_signal):
    """
    Plots the Xic, modulated signal and residual signal for an specific m/z in 2 different subplots 

    Parameters
    ----------
    xic
    modulated_signal
    residual_signal

    Returns
    -------
    None.

    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # First subplot: original and modulated signal
    axs[0].plot(rts, xic, label='XIC original', linewidth=0.8, color='grey')
    axs[0].plot(rts, modulated_signal, label='Modulated signal', linestyle='--', color='orange')
    axs[0].set_ylabel("Intensity")
    axs[0].set_title(f"XIC and Modulated Signal for m/z = {target_mz}")
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot: residual signal
    axs[1].plot(rts, residual_signal, label='Residual signal', color='green', linewidth=0.8)
    axs[1].set_xlabel("Retention time (s)")
    axs[1].set_ylabel("Intensity")
    axs[1].set_title("Residual Signal")
    axs[1].legend()
    axs[1].grid(True)
    
    #Para que tengan la misma escala
    all_values = np.concatenate([xic, modulated_signal, residual_signal])
    y_min, y_max = np.min(all_values), np.max(all_values)

    # Aplicar la misma escala
    axs[0].set_ylim(y_min, y_max)
    axs[1].set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()
    
def plot_modulated_signal(rts, target_mz, modulated_signal):
    plt.figure(figsize=(10, 6))
    plt.plot(rts, modulated_signal, label='Modulated signal', linestyle='--', color='orange')
    plt.xlabel("Retention time (s)")
    plt.ylabel("Intesity")
    plt.title(f"Modulated signal for m/z = {target_mz}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residual_signal(rt_array, target_mz, residual_signal):
    """
    Plots only the residual signal versus retention time.

    Parameters
    ----------
    rt_array : array-like
        Retention times.
    residual_signal : array-like
        Residual signal after modulation removal.
    target_mz : float, optional
        Target m/z for reference in the title.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rt_array[:40], residual_signal[:40], label='Residual signal', color='blue', linewidth=0.9)

    plt.xlabel("Retention time (s)")
    plt.ylabel("Residual intensity")
    title = "Residual Signal"
    if target_mz is not None:
        title += f" for m/z = {target_mz}"
    plt.title(title)
    
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()