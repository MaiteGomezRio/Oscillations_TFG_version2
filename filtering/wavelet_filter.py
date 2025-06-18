# -*- coding: utf-8 -*-
import pywt
import numpy as np
import matplotlib.pyplot as plt

def plot_original_and_residual(rts, target_mz, xic, residual_signal):
    plt.figure(figsize=(12, 6))
    plt.plot(rts, xic, label="XIC original", color='gray')
    plt.plot(rts, residual_signal, label="Residual signal", color='green')
    plt.title(f"XIC, and residual for m/z = {target_mz}")
    plt.xlabel("Retention time (s)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def extract_xic(rts, mz_array, intensity_array, target_mz, mz_tol=0.1):
    xic = []
    for mzs, intensities in zip(mz_array, intensity_array):
        mask = np.abs(mzs - target_mz) <= mz_tol
        xic.append(np.sum(intensities[mask]) if np.any(mask) else 0)
    return np.array(xic)


def wavelet_denoise(signal, wavelet='db4', level=4, threshold_method='universal'):
    """
    Filtra una señal eliminando oscilaciones periódicas mediante wavelet denoising.

    Parameters:
        signal (np.ndarray): La señal original.
        wavelet (str): Tipo de wavelet (e.g., 'db4', 'sym5').
        level (int): Nivel de descomposición.
        threshold_method (str): Método de umbral ('universal', 'minimax', etc.)

    Returns:
        np.ndarray: Señal filtrada (solo picos).
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # estimación del ruido 
    if threshold_method == 'universal':
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))*2.5
    else:
        sigma

    # Aplica umbral suave en todos los niveles de detalle
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    
    filtered_signal = pywt.waverec(coeffs, wavelet)

    return filtered_signal[:len(signal)] 



def correct_oscillations_wavelet(rts, mz_array, intensity_array, target_mz, mz_tol=0.01):
    xic=extract_xic(rts, mz_array, intensity_array, target_mz)
    residual_signal = wavelet_denoise(xic, wavelet='db4', level=4)
    #residual_signal = xic - modulated_signal
    return xic, residual_signal
