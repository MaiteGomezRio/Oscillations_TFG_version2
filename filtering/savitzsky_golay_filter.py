# utils/filters.py
from scipy.signal import savgol_filter
import numpy as np

def apply_savgol_filter(intensities, window_length, filter_order):
    """
    Aplica un filtro Savitzky-Golay para suavizar la señal.

    Args:
        intensities (array-like): Intensidades de la señal.
        window_length (int): Tamaño de ventana (debe ser impar).
        filter_order (int): Orden del polinomio del filtro.

    Returns:
        np.array: Intensidades suavizadas.
    """
    if len(intensities) > window_length:
        smoothed = savgol_filter(intensities, window_length, filter_order)
    else:
        smoothed = np.array(intensities)  # Sin suavizado si es muy corta

    return smoothed
