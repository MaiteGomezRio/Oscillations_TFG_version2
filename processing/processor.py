# processing/processor.py
import pyopenms as oms
import os
import time
import numpy as np
import pandas as pd
#from collections import Counter
from processing.corrector_3 import correct_oscillations, plot_all, plot_modulated_signal, plot_residual_signal, build_xic, apply_polynomial_regression
from utils.io_utils import convert_mzxml_2_mzml
from validation.signal_validator2 import plot_corrected_signal_bpc, compare_bpc_values, compare_frequencies_over_rt,  build_variable_frequency_sine_and_plot_residual, verify_residual_signal, debug_correccion
#from simulated_corrector2_tryFile import plot_signal, correct_oscillations_per_bin, detect_oscillating_mzs, plot_tic_comparison, export_tic_signals_to_csv, export_xic_signals_combined_csv
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from processing.notch_filter import notch_filter, plot_xic_filtered
from processing.wavelet_filter import correct_oscillations_wavelet, plot_original_and_residual
from processing.gaussian_filter import apply_gaussian_filter
from utils.fft_utils2 import local_frequencies_with_fft

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
    
def correct_spectra(input_map, tic_original, residual_signal):
    """
    TIC
    
    Corrects the spectra with TIC

    Parameters
    ----------
    input_map : TYPE
        DESCRIPTION.
    tic_original : TYPE
        DESCRIPTION.
    residual_signal : TYPE
        DESCRIPTION.

    Returns
    -------
    corrected_map : TYPE
        DESCRIPTION.

    """
    #Creo un nuevo MSExperiment corregido
    corrected_map = oms.MSExperiment()
    adjusted_intensities=[]
    scale_factors=[]
    for i, spectrum in enumerate(input_map):
        new_spectrum = oms.MSSpectrum(spectrum)  # clonar espectro original
        mzs, intensities = new_spectrum.get_peaks()
         
        # Escalo todas las intensidades por el factor residual/TIC original para que se aplique la 
        #corrección a cada espectro
        #tic_threshold = 0.01 * max(tic_original)  # 1% del TIC máximo
        #TODO--si la residual signal es mayor que 
        if tic_original[i] > 0:
            scale_factor = residual_signal[i] / tic_original[i]
            scaled_intensities=intensities * scale_factor
            
        else:
            scale_factor=1.0 # evitar división por cero
            scaled_intensities=intensities
            
        
        new_spectrum.set_peaks((mzs, scaled_intensities))
        corrected_map.addSpectrum(new_spectrum)
    
    return corrected_map

def correct_spectra2(input_map, target_mz_list, rts, residual_signals, mz_tol=0.1):
    """
    MZ WITH MODULATED SIGNAL 

    Parameters
    ----------
    input_map : TYPE
        DESCRIPTION.
    target_mz_list : TYPE
        DESCRIPTION.
    rts : TYPE
        DESCRIPTION.
    residual_signals : TYPE
        DESCRIPTION.
    mz_tol : TYPE, optional
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    corrected_map : TYPE
        DESCRIPTION.

    """
    # para mantener el input_map intacto
    corrected_map = oms.MSExperiment()
    
    # Copiar espectros para no modificar input_map
    for spec in input_map:
        new_spec = oms.MSSpectrum(spec)  # constructor que copia los datos
        corrected_map.addSpectrum(new_spec)

    # Interpolar de señales residuales por m/z
    interpoladores = {
        target_mz: lambda rt, res=residual_signals[target_mz]: float(np.interp(rt, rts, res))
        for target_mz in target_mz_list
    }
    
    for spec in corrected_map:
        
        rt = spec.getRT()
        mzs, intensities = spec.get_peaks()
        intensities = np.array(intensities)

        for i in range(len(mzs)):
            for target_mz in target_mz_list:
                if abs(mzs[i] - target_mz) <= mz_tol:
                    residual = interpoladores[target_mz](rt)
                    #print(f"Match at RT={rt:.2f}s | m/z={mzs[i]:.4f} ~ {target_mz:.4f} -> Intensity set to {residual:.2f}")
                    intensities[i] = residual
                    #print(f"Intensity at Mz: {i}    is: {intensities[i]:.2f}")

        spec.set_peaks((mzs, intensities))

    return corrected_map

def correct_spectra3(input_map, target_mz_list, rts, residual_signals, mz_tol=0.1):
    """
    GAUSSIAN FILTER
    Applies correction to spectra. 
    En las zonas donde no hay picos hace la resta original-residual,  y en los que si sustituye la residual

    Parameters
    ----------
    input_map : TYPE
        DESCRIPTION.
    target_mz_list : TYPE
        DESCRIPTION.
    rts : TYPE
        DESCRIPTION.
    residual_signals : TYPE
        DESCRIPTION.
    mz_tol : TYPE, optional
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    Corrected map

    """
    corrected_map = oms.MSExperiment()
    rt_to_index = {round(rt, 6): i for i, rt in enumerate(rts)}
    
    
    #calculo umbrales para cada mz 
    peak_factor=3.0
    thresholds = {}

    for mz, signal in residual_signals.items():
        media = np.mean(signal)
        desviacion = np.std(signal)
        threshold = media + peak_factor * desviacion
        thresholds[mz] = threshold
    
    
    for spec in input_map:
        new_spec = oms.MSSpectrum(spec)  # copia del espectro original
        mzs, intensities = new_spec.get_peaks()
        rt = round(spec.getRT(), 6)
        
        if rt not in rt_to_index:
            continue
        
        idx = rt_to_index[rt]
        
        for target_mz in target_mz_list:
           mask = (mzs >= target_mz - mz_tol) & (mzs <= target_mz + mz_tol)
           if np.any(mask):
               # Sustituye las intensidades de ese m/z por el valor residual corregido
               original_intensity = intensities[mask]
               filtered_intensity = residual_signals[target_mz][idx]
               threshold=thresholds[target_mz]
               
               if filtered_intensity>=threshold:
                   corrected_intensity=filtered_intensity
               else:
                   corrected_intensity=original_intensity-filtered_intensity
               
                
               intensities[mask]=corrected_intensity
               
        new_spec.set_peaks((mzs, intensities))
        corrected_map.addSpectrum(new_spec)
    
    return corrected_map

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

def obtain_freq_from_signal(rt_array, mz_array, intensity_array, window_size=70, mz_ref=922.098):
    """
    Obtains frequency from reference mz

    Parameters
    ----------
    rt_array : TYPE
        DESCRIPTION.
    mz_array : TYPE
        DESCRIPTION.
    intensity_array : TYPE
        DESCRIPTION.
    window_size : TYPE, optional
        DESCRIPTION. The default is 70.
    mz_ref : TYPE, optional
        DESCRIPTION. The default is 922.098.

    Returns
    -------
    rt_freqs : TYPE
        DESCRIPTION.
    local_freqs : TYPE
        DESCRIPTION.
    phase : TYPE
        DESCRIPTION.

    """
    xic=build_xic(mz_array, intensity_array, rt_array, target_mz=mz_ref)
    sampling_interval = np.mean(np.diff(rt_array))
    rt_freqs, local_freqs_ref = local_frequencies_with_fft(xic, rt_array, window_size, sampling_interval)
    phase_ref=apply_polynomial_regression(rt_array, rt_freqs, local_freqs_ref)

    return local_freqs_ref, phase_ref


def process_file(file_path, save_as):
    
    start_time=time.time()
    input_map = oms.MSExperiment()

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".mzxml":
        mzml_file_path = convert_mzxml_2_mzml(file_path)
        print("Converting to mzML...")
        time.sleep(3)
        if not os.path.exists(mzml_file_path):
            raise RuntimeError("Error: file not found")
        else:
            print("Loading mzML file...")
            oms.MzMLFile().load(mzml_file_path, input_map)
    else:
        oms.MzMLFile().load(file_path, input_map)

    original_spectra = []
    mz_array = []
    intensity_array=[]
    rts = []#en segundos
    tic_original = []
   
    print("Processing file....")
    print(">>> Corrigiendo")
    
    
    # 1. Obtengo primero la información de la señal original 
    for spectrum in input_map:
        original_spectra.append(spectrum)
        mzs, intensities = spectrum.get_peaks()
        mz_array.append(mzs)
        intensity_array.append(intensities)
        rt = spectrum.getRT()
        rts.append(rt)
        tic_original.append(np.sum(intensities))
    
    
    # 2. Corrijo las oscilaciones
        #2.1 Genero la onda de ref
        #2.2 Obtengo la onda de mz2 (sin el pico)
        #2.3 Le resto a la original (mz2) la generada (mz1) para que se me quede solo el pico
        
    
    #Obtengo la frecuencia de la señal de ref para mz=922.098
    #ref_mz=922.098
    local_freqs_ref, phase_ref = obtain_freq_from_signal(rts, mz_array, intensity_array)
    
    
    #Corrijo localmente
    target_mz_list=[60.07, 65.1, 95.1, 96.085, 110.1173, 922.098]
    xic_signals = {}
    modulated_signals = {}#Dict[target_mz: float, modulated: np.ndarray]
    residual_signals = {}#Dict[target_mz: float, residual: np.ndarray]

    for target_mz in target_mz_list:
        #xic, residual_signal = apply_gaussian_filter(rts, mz_array, intensity_array, target_mz)
        xic, modulated_signal, residual_signal=correct_oscillations(rts, mz_array, intensity_array, phase_ref, local_freqs_ref, target_mz)
        plot_all(rts, target_mz, xic, modulated_signal, residual_signal)
        #plot_original_and_residual(rts, target_mz, xic, residual_signal)
        #plot_modulated_signal(rts, target_mz, modulated_signal)
        xic_signals[target_mz] = xic
        modulated_signals[target_mz] = modulated_signal
        residual_signals[target_mz] = residual_signal
    
    
    #[DEBUG] Data exported to csv
    """
    FOR TIC
    output_csv_path="C:/Users/maipa/repos/oscilations_TFG_repository/FILES/MS_files/TIC_COMPARISON.csv"
    export_tic_signals_to_csv(rts, tic_original, modulated_signal, residual_signal, output_csv_path)
    """
    #FOR EACH MZ
    output_path="C:/Users/maipa/repos/oscilations_TFG_repository/FILES/MS_files/PRUEBAS_ultimas/XIC_prueba4.csv"
    export_xic_signals_combined_csv(rts, xic_signals, modulated_signals, residual_signals, output_path)
    
    
    
    
    # 3. Aplico el cambio a los espectros
    corrected_map=correct_spectra2(input_map, target_mz_list, rts, residual_signals)
    #corrected_map=correct_spectra3(input_map, target_mz_list, rts, residual_signals)
    
    
    #Calculo tiempo de ejecución
    end_time=time.time()
    time_elapsed=end_time-start_time
    
    print("<<< Corrección terminada") 
    print(f"Execution time: {time_elapsed:.3f}")
    
    # 4. Guardo los espectros corregidos
    #input_map.setSpectra(corrected_map)#para susituir el original file
    oms.MzMLFile().store(save_as, corrected_map)
    print(f"Corrected file saved: {save_as}")

