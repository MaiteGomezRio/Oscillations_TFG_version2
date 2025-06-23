

import numpy as np
from scipy.integrate import cumulative_trapezoid
from utils.fft_utils2 import local_frequencies_with_fft
from utils.filters import apply_savgol_filter
import matplotlib.pyplot as plt
import pandas as pd

def build_xic(mz_array, intensity_array, rt_array, target_mz, mz_tol=0.1):
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

def get_amplitude(target_mz, xic, rt_array, local_freqs, sampling_interval):
    
    
    local_amplitudes=[]
    
    
    for i, freq in enumerate (local_freqs):
        
        if freq<=0:
            continue
        
        period=int(1/(freq*sampling_interval))
        
        if period<3: #para que sea al menos 3 si no es muy pequeña
            continue
        
        center=i*int(len(xic) / len(local_freqs))
        start=int(max(0, center-period/2))
        end=int(min(len(xic), center+period/2))
        
        
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
    
def correct_oscillations(rt_array, mz_array, intensity_array, phase_ref, local_freqs_ref, target_mz, window_size=70):
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
    
    #TODO-> SHOULD BE CONSTANT FOR ALL MZ (EXTRACTED FROM MZ 922.098) BUT ADAPT WITH A 
    #POLYNOMIAL REGRESSION TO THE SIGNAL
    sampling_interval = np.mean(np.diff(rt_array))
    #rt_freqs, local_freqs = local_frequencies_with_fft(xic, rt_array, window_size, sampling_interval)
    #phase=apply_polynomial_regression(rt_array, rt_freqs, local_freqs)
    
    
    # 3. Amplitude at each m/z
    amplitude=get_amplitude(target_mz, xic, rt_array, local_freqs_ref, sampling_interval)
    #print(f"Amplitude for m/z: {target_mz} is: {amplitude}")
    
    # 4. Creation of the modulated signal
    modulated_signal=generate_modulated_signal(amplitude, phase_ref)
    offset=np.median(xic)
    #offset = apply_savgol_filter(xic, window_length=70, filter_order=2)
    #modulated_signal+=offset
    
    # 5. Computation of the residual/final signal
    residual_signal=xic-np.median(modulated_signal)
    #esidual_signal = np.where(modulated_signal > xic, modulated_signal - xic,
                           #xic - modulated_signal)
    #residual_signal_adjusted=np.clip(residual_signal, 0, None)
    
    #[DEBUG]
    
    """print("Mz: 65.1")
    if target_mz==65.1:
        residual_df = pd.DataFrame({
            "RT (s)": rt_array[:40],
            "Original": xic[:40],
            "Modulated": modulated_signal[:40],
            "Residual: ": residual_signal[:40]
        })
        # Imprimir todos los valores
        print(residual_df.to_string(index=False))
    """
    
    return xic, modulated_signal, residual_signal
    
    