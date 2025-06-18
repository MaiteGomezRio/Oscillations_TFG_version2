import numpy as np
from scipy.signal import iirnotch, filtfilt, periodogram
import matplotlib.pyplot as plt
from utils.fft_utils2 import local_frequencies_with_fft

def plot_xic_filtered(rts, xic_original, corrected_signal, target_mz):
    """
    Plotea la señal XIC antes y después de aplicar el filtro notch.

    Parameters:
    - rt: np.array, vector de tiempo de retención
    - intensidad_original: np.array, señal original
    - intensidad_filtrada: np.array, señal después de filtrar
    - mz: float, valor de m/z para etiquetar el gráfico
    - titulo_extra: str, texto adicional para el título (opcional)
    """
    plt.figure(figsize=(12, 5))
    plt.plot(rts, xic_original, label="Original", alpha=0.6, linewidth=0.8)
    plt.plot(rts, corrected_signal, label="Filtrada", linewidth=0.8)
    plt.title(f"XIC para m/z = {target_mz:.4f}")
    plt.xlabel("Tiempo de Retención (RT)")
    plt.ylabel("Intensidad")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_main_local_freq(xic, rts, window_size):
    sampling_interval = np.median(np.diff(rts))  # segundos por muestra
    _, freqs = local_frequencies_with_fft(xic, rts, window_size, sampling_interval)
    freqs = freqs[~np.isnan(freqs)]  # elimina NaNs si hay
    if len(freqs) == 0:
        return None  # o algún valor por defecto
    return np.median(freqs)



def notch_filter(xic, rts, q=30.0):
    """
    Aplica un filtro notch adaptativo a una señal para eliminar oscilaciones periódicas.
    """
    fs = 1 / np.median(np.diff(rts))  # frecuencia de muestreo
    #f, pxx = periodogram(xic, fs)
    
    main_local_freq=calculate_main_local_freq(xic, rts, window_size=30)

    print(f"[DEBUG] f0 = {main_local_freq:.4f} Hz, fs = {fs:.4f} Hz, Nyquist = {fs/2:.4f} Hz")
    #print(f"[Filtro] Frecuencia dominante detectada: {main_local_freq:.4f} Hz")


    # Paso 2: diseñar y aplicar filtro notch
    b, a = iirnotch(main_local_freq, q, fs)#we get filter coefficients
    corrected_signal = filtfilt(b, a, xic)#applies filtering to signal
    
    
    return corrected_signal

