# processing/processor.py
import pyopenms as oms
import os
import time
import numpy as np
import pandas as pd
<<<<<<< HEAD
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import defaultdict#used to extract all mzs with signal

from processing.corrector_3 import correct_oscillations, build_xic, apply_polynomial_regression
from utils.io_utils import convert_mzxml_2_mzml
#from filtering.notch_filter import notch_filter, plot_xic_filtered
#from filtering.wavelet_filter import correct_oscillations_wavelet, plot_original_and_residual
#from filtering.gaussian_filter import apply_gaussian_filter
=======
#from collections import Counter
from processing.corrector_3 import correct_oscillations, plot_all, plot_modulated_signal, plot_residual_signal, build_xic, apply_polynomial_regression
from utils.io_utils import convert_mzxml_2_mzml
#from validation.signal_validator2 import plot_corrected_signal_bpc, compare_bpc_values, compare_frequencies_over_rt,  build_variable_frequency_sine_and_plot_residual, verify_residual_signal, debug_correccion
#from simulated_corrector2_tryFile import plot_signal, correct_oscillations_per_bin, detect_oscillating_mzs, plot_tic_comparison, export_tic_signals_to_csv, export_xic_signals_combined_csv
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from filtering.notch_filter import notch_filter, plot_xic_filtered
from filtering.wavelet_filter import correct_oscillations_wavelet, plot_original_and_residual
from filtering.gaussian_filter import apply_gaussian_filter
>>>>>>> 7d7ea025b752bc51bb8e4d5d55ba7966bc4ace72
from utils.fft_utils2 import local_frequencies_with_fft
from validation.validator import plot_all, export_xic_signals_combined_csv


    
def correct_spectra_with_tic(input_map, tic_original, residual_signal):
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

def correct_spectra_gaussian_filter(input_map, target_mz_list, rts, residual_signals, mz_tol=0.1):
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

def correct_spectra(input_map, target_mz_list, rts, residual_signals, mz_tol=0.1):
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
    corrected_map.setMetaData(input_map.getMetaData())
    
    
    mz_keys = np.array(sorted(residual_signals.keys()))#las mismas que target_mz_list
    
    # Obtengo los mzs e intensidades del file original
    for i, spectrum in enumerate(input_map):
        mzs, original_intensities = spectrum.get_peaks()
        mzs = np.array(mzs)
        corrected_intensities = np.zeros_like(mzs)
        #le paso el input_map para poder crear el corrected

        for j, mz in enumerate(mzs):
            # Busca m/z corregida más cercana dentro de la tolerancia
            idx = np.where(np.abs(mz_keys - mz) <= mz_tol)[0]
            if idx.size > 0:
                closest_mz = mz_keys[idx[np.argmin(np.abs(mz_keys[idx] - mz))]]
                corrected_intensities[j] = residual_signals[closest_mz][i]
            else:
                corrected_intensities[j] = original_intensities[j]#si no encuentra esa mz corregida que deje la original

        # Reconstruye espectro corregido
        new_spectrum = oms.MSSpectrum()
        new_spectrum.set_peaks((mzs, corrected_intensities))
        new_spectrum.setRT(spectrum.getRT())
        new_spectrum.setMSLevel(spectrum.getMSLevel())
        new_spectrum.setDriftTime(spectrum.getDriftTime())
        new_spectrum.setPrecursors(spectrum.getPrecursors())
        new_spectrum.setInstrumentSettings(spectrum.getInstrumentSettings())
        new_spectrum.setAcquisitionInfo(spectrum.getAcquisitionInfo())
        new_spectrum.setType(spectrum.getType())

        corrected_map.addSpectrum(new_spectrum)

    return corrected_map

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

def extract_detected_mzs(rt_array, mz_array, intensity_array, intensity_threshold=1.0, decimal_precision=2):
    """
    Extracts all mzs with signal 

    Parameters
    ----------
    rt_array : TYPE
        array with rts from the original signal
    mz_array : list
        list of arrays of m/z 
    intensity_array : list
        list of arrays of intensities

    Returns
    -------
    List of mzs with an intensity higher than the treshold established

    """
    mz_set = set()

    
    for mzs, intensities in zip(mz_array, intensity_array):#para cada espectro sus mzs y sus intensidades 
        for mz, intensity in zip(mzs, intensities):#en cada espectro mz e intensidad
            if intensity >= intensity_threshold:
                mz_rounded = round(mz, decimal_precision)
                mz_set.add(mz_rounded)

    return sorted(mz_set)

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
    
        #2.1 Obtengo la frecuencia de la señal de ref para mz=922.098
        #ref_mz=922.098
        
    local_freqs_ref, phase_ref = obtain_freq_from_signal(rts, mz_array, intensity_array)
    
    
        #2.2 Corrijo localmente
        #target_mz_list=[60.07, 65.1, 95.1, 96.085, 110.1173, 922.098]
        
    target_mz_list=extract_detected_mzs(rts, mz_array, intensity_array)
    
    print(f"TARGET MZS DETECTED: {target_mz_list}")
    
    
    xic_signals = {}
    modulated_signals = {}#Dict[target_mz: float, modulated: np.ndarray]
    residual_signals = {}#Dict[target_mz: float, residual: np.ndarray]

    for target_mz in target_mz_list:
        
        xic, modulated_signal, residual_signal=correct_oscillations(rts, mz_array, intensity_array, phase_ref, local_freqs_ref, target_mz)
        #plot_all(rts, target_mz, xic, modulated_signal, residual_signal)
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
    output_path="C:/Users/maipa/repos/oscilations_TFG_repository/FILES/MS_files/PRUEBAS_ultimas/XIC_prueba6.csv"
    export_xic_signals_combined_csv(rts, xic_signals, modulated_signals, residual_signals, output_path)
    
    
    # 3. Aplico el cambio a los espectros
    corrected_map=correct_spectra(input_map, target_mz_list, rts, residual_signals)
    
    
    #Calculo tiempo de ejecución
    end_time=time.time()
    time_elapsed=end_time-start_time
    
    print("<<< Corrección terminada") 
    print(f"Execution time: {time_elapsed:.3f}")
    
    # 4. Guardo los espectros corregidos
    #input_map.setSpectra(corrected_map)#para susituir el original file
    oms.MzMLFile().store(save_as, corrected_map)
    print(f"Corrected file saved: {save_as}")

