import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from utils.filters import apply_savgol_filter
from scipy.integrate import cumulative_trapezoid


def obtain_amplitudes(mzs, intensities, bin_size):
    min_mz = np.min(mzs)
    max_mz = np.max(mzs)
    bins = np.arange(min_mz, max_mz + bin_size, bin_size)
    amplitudes, _ = np.histogram(mzs, bins, weights=intensities)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return amplitudes, bin_centers

def extract_amplitudes_at_mz(input_map, bin_size, target_mz, window_length=15, filter_order=3, plot=False):
    """
    Extrae amplitudes en un m/z específico a lo largo del RT de un archivo mzML cargado con pyOpenMS.
    
    Args:
        input_map: objeto MSExperiment cargado con pyOpenMS.
        target_mz (float): m/z objetivo.
        bin_size (float): tamaño del bin para histogramas.
        window_length (int): ventana del filtro Savitzky-Golay.
        filter_order (int): orden del polinomio para el filtro.
        plot (bool): si True, genera gráfico.
        
    Returns:
        rt_array (np.array): tiempos de retención.
        amplitude_array (np.array): amplitudes extraídas en target_mz.
    """
    amplitude_list = []
    rt_list = []
    mz_tol=0.09

    for spectrum in input_map:
        mzs, intensities = spectrum.get_peaks()
        rt = spectrum.getRT()

        # Suaviza intensidades
        smoothed_intensities = apply_savgol_filter(intensities, window_length, filter_order)
        
        
        # Calcula histogramas (bins)
        amplitudes, bin_centers = obtain_amplitudes(mzs, smoothed_intensities, bin_size)


        # Busca el índice del mz más cercano
        idx_closest = np.argmin(np.abs(mzs - target_mz))
        mz_closest = mzs[idx_closest]


        # Si está dentro de la tolerancia, guarda su intensidad
        if np.abs(mz_closest - target_mz) < mz_tol:
            amplitude = smoothed_intensities[idx_closest]
        else:
            amplitude = 0.0

        amplitude_list.append(amplitude)
        rt_list.append(rt)

    rt_array = np.array(rt_list)
    amplitude_array = np.array(amplitude_list)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(rt_array, amplitude_array, color='darkred')
        plt.xlabel("Retention Time (RT)")
        plt.ylabel(f"Amplitude at m/z {target_mz}")
        plt.title(f"Amplitudes over RT at m/z ≈ {target_mz}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return rt_array, amplitude_array, bin_size

def calculate_freq(xic, sampling_interval=1.0, plot_spectrum=False):
    """
    Estima la frecuenciaa dominante usando fft
    
    Parameters
    ----------
    intensities : signal intensities
        DESCRIPTION.
    sampling_interval : TYPE, optional
        DESCRIPTION. The default is 1.0.
    plot_spectrum : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    fft_freqs : TYPE
        DESCRIPTION.
    fft_magnitude : TYPE
        DESCRIPTION.
    main_freq : TYPE
        DESCRIPTION.

    """
    centered_signal = xic - np.mean(xic)
    fft_result = fft(centered_signal)
    freqs = np.fft.fftfreq(len(centered_signal), d=sampling_interval)
    
    #Solo frecuencias positivas
    pos_mask = freqs > 0
    fft_freqs = freqs[pos_mask]
    fft_magnitude = np.abs(fft_result[pos_mask])
    
    main_freq = fft_freqs[np.argmax(fft_magnitude)]

    if plot_spectrum:
        plt.figure(figsize=(10, 4))
        plt.plot(fft_freqs, fft_magnitude, color='darkgreen')
        plt.title("Espectro de Frecuencia (FFT)")
        plt.xlabel("Frecuencia (ciclos/minuto)")
        plt.ylabel("Magnitud")
        plt.tight_layout()
        plt.show()
        

    return fft_freqs, fft_magnitude, main_freq

def local_frequencies_with_fft(xic, rts, window_size, sampling_interval):
    freqs = []
    times = []
    step = window_size // 2

    for i in range(0, len(xic) - window_size, step):
        segment = xic[i:i+window_size]
        rt_segment = rts[i:i+window_size]
        
        _, _, dom_freq = calculate_freq(segment, sampling_interval)
        
        freqs.append(dom_freq)
        times.append(np.mean(rt_segment))

    return np.array(times), np.array(freqs)

def apply_polynomial_regression(rts, rt_freqs, local_freqs, freq_deg=2):
    
    """
    Applies polynomial regression for frequency to match correctly the one of the original signal
    """
    rts = np.array(rts)
    t = (rts - rts[0])
    
    freq_interp = np.interp(rts, rt_freqs, local_freqs)
    fit=np.polyfit(rts, freq_interp, freq_deg)#ajusta el polinomio a los datos
    freq_poly = np.poly1d(fit)
    f_t = freq_poly(t)#frecuencia suavizada en cada punto t

    # 3. Calcular fase acumulada φ(t) con integración
    phase = 2 * np.pi * cumulative_trapezoid(f_t, t, initial=0)
    
    return phase