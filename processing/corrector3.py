# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import cumulative_trapezoid
from utils.fft_utils2 import local_frequencies_with_fft
from utils.filters import apply_savgol_filter
import matplotlib.pyplot as plt
import pandas as pd

def build_xic(mz_array, intensity_array, rt_array, target_mz, mz_tol=0.1):#TODO REVISAR 
    """
    Builds XIC for a given m/z value
    
    Returns: Array of intensities along RT for that given m/z
    """
    xic = []
    for mzs, intensities in zip(mz_array, intensity_array):
        is_in_tol = np.abs(mzs - target_mz) < mz_tol
        if np.any(is_in_tol):
            xic.append(np.sum(intensities[is_in_tol]))
        else:
            xic.append(0.0)
            
    return np.array(xic)

def apply_polynomial_regression(rts, rt_freqs, local_freqs, freq_deg=2):
    rts = np.array(rts)
    t = (rts - rts[0])
    
    freq_interp = np.interp(rts, rt_freqs, local_freqs)
    fit=np.polyfit(rts, freq_interp, freq_deg)#ajusta el polinomio a los datos
    freq_poly = np.poly1d(fit)
    f_t = freq_poly(t)#frecuencia suavizada en cada punto t

    # 3. Calcular fase acumulada φ(t) con integración
    phase = 2 * np.pi * cumulative_trapezoid(f_t, t, initial=0)
    
    return phase 

def extract_amplitude_at_mz(rt_array,target_mz, xic, local_freqs, cycles_per_bin=1):
    
    """
    Divides the signal in local freq bins throughout RT and estimates an amplitude for each bin
    ----------
    Parameters: 
        - cycles_per_bin: min number of cycles per bin
    """
    #para estimar freq dominante
    is_not_nan=np.isnan(local_freqs)==False
    valid_freqs = local_freqs[is_not_nan]

    dominant_freq = np.median(valid_freqs)
    bin_width = (1.0 / dominant_freq) * cycles_per_bin

    num_bins = int((rt_array[-1] - rt_array[0]) / bin_width)
    bin_edges = np.linspace(rt_array[0], rt_array[-1], num_bins + 1)

    amplitudes = []

    for i in range(num_bins):
        start_rt = bin_edges[i]
        end_rt = bin_edges[i + 1]
        bin_mask = (rt_array >= start_rt) & (rt_array < end_rt)
        if np.sum(bin_mask) < 3:#podría poner 5 
            continue
        xic_bin = xic[bin_mask]
        amp = np.max(xic_bin) - np.min(xic_bin)
        amplitudes.append(amp)
    
    main_amplitude=np.median(amplitudes)
    
    # Offset para centrar la onda en la señal original
    offset_curve = apply_savgol_filter(xic, window_length=70, filter_order=2)
    
    #(f"Main amplitude for mz {target_mz} = {main_amplitude}")
    return main_amplitude, offset_curve

def get_amplitude(target_mz, xic, rt_array, local_freqs, sampling_interval):
    
    
    #tamaño de la ventana de muestreo
    window_size=int((len(xic)-1)/len(rt_array))
    
    local_amplitudes=[]
    step=window_size/2#centro de la ventana
    
    for i, freq in enumerate (local_freqs):
        period=int(1/(freq*sampling_interval))
        center=i*step
        start=int(max(0, center-period/2))
        end=int(min(len(xic), center+period/2))
        
        if end-start<3: #podria poner más
            continue
        window=xic[start:end]
        local_amplitude=np.max(window)-np.min(window)
        local_amplitudes.append(local_amplitude)
        
    amplitude=np.median(local_amplitudes)/2
    
    return amplitude
    
def generate_modulated_signal(amplitude, phase):
    """
    Generates the signal that is going to substract the original one at each m/z so the oscillations can be corrected

    Returns:
        - mmodulated_signal: 
    -------
    
    """
    
    modulated_signal = amplitude * np.sin(phase) 
    
    return modulated_signal

def plot_all(rts, target_mz, xic, modulated_signal, residual_signal):
    """
    Plots the Xic, modulated signal and residual signal for an specific m/z

    Parameters
    ----------
    xic
    modulated_signal
    residual_signal

    Returns
    -------
    None.

    """
    plt.figure(figsize=(10, 6))
    plt.plot(rts, xic, label='XIC original', linewidth=0.8, color='grey')
    plt.plot(rts, modulated_signal, label='Modulated signal', linestyle='--', color='orange')
    plt.plot(rts, residual_signal, label='Residual signal', color='green', linewidth=0.8)

    plt.xlabel("Retention time (s)")
    plt.ylabel("Intesity")
    plt.title(f"XIC, Modulated signal and residual for m/z = {target_mz}")
    plt.legend()
    plt.grid(True)
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
    
def correct_oscillations(rt_array, mz_array, intensity_array, target_mz, window_size=70):
    """
    For each target mz, generates a modulated signal that is created with the frequency
    and amplitude of the signal. 
    
    Parameters:
        - 

    Returns: A residual signal for each mz that is the result of the original xic - modulated signal
    -------

    """
    #1. Extract XIC from original signal (intensities for each RT at target_mz)
    xic=build_xic(mz_array, intensity_array, rt_array, target_mz)
    

    #2. Frequency with polynomial regression
    sampling_interval = np.mean(np.diff(rt_array))
    rt_freqs, local_freqs = local_frequencies_with_fft(xic, rt_array, window_size, sampling_interval)
    phase=apply_polynomial_regression(rt_array, rt_freqs, local_freqs)
    
    
    # 3. Amplitude at each m/z
    #amplitude, offset_curve=extract_amplitude_at_mz(rt_array, target_mz, xic, local_freqs)
    amplitude=get_amplitude(target_mz, xic, rt_array, local_freqs, sampling_interval)
    #print(f"Amplitude for m/z: {target_mz} is: {amplitude}")
    
    # 4. Creation of the modulated signal
    modulated_signal=generate_modulated_signal(amplitude, phase)
    offset=np.median(xic)#baseline
    modulated_signal_with_offset=modulated_signal + offset
    
    # 5. Computation of the residual/final signal
    residual_signal=xic-modulated_signal_with_offset
    #residual_signal_adjusted=np.clip(residual_signal, 0, None)
    
    if target_mz==65.1:
        residual_df = pd.DataFrame({
            "RT (s)": rt_array[:40],
            "Residual Signal": residual_signal[:40]
        })
        # Imprimir todos los valores
        print(residual_df.to_string(index=False))
    
    
    return xic, modulated_signal_with_offset, residual_signal
    
    