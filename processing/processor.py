# processing/processor.py
import pyopenms as oms
import os
import time
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import defaultdict#used to extract all mzs with signal

from processing.corrector_3 import correct_oscillations, build_xic, apply_polynomial_regression
from utils.io_utils import convert_mzxml_2_mzml
#from filtering.notch_filter import notch_filter, plot_xic_filtered
#from filtering.wavelet_filter import correct_oscillations_wavelet, plot_original_and_residual
#from filtering.gaussian_filter import apply_gaussian_filter
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

def detect_oscillating_mzs(rts, mz_array, intensity_array, base_target_mzs, phase_ref, local_freqs_ref, intensity_threshold=100, osc_treshold=0.1, max_mzs_to_check=500):
    detected_mzs=extract_detected_mzs(rts, mz_array, intensity_array)
    auto_selected_mzs=[]
    
    for mz in detected_mzs:
        if mz in base_target_mzs:
            continue
        try:
            xic, modulated_signal, _=correct_oscillations(rts, mz_array, intensity_array, phase_ref, local_freqs_ref, mz)
            intensity_importance=np.var(modulated_signal)/(np.var(xic)+1e-8)
            if intensity_importance > osc_treshold:
                auto_selected_mzs.append(mz)
        except Exception as e:
            print(f"[!] Error analyzing m/z {mz}:{e}")
            
        if len(auto_selected_mzs) >= max_mzs_to_check:
            break #porque si no hay sobrecarga de memoria/CPU 
    
    return auto_selected_mzs
        
def correct_spectra(input_map, target_mz_list, rts, residual_signals, mz_tol=0.1, threshold=0.0012):
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
    
    
    n_corrected = 0
    
    
    mz_keys = np.array(sorted([float(k) for k in residual_signals.keys()]))
    
    # Calcula el umbral relativo
    all_vals = np.concatenate([np.array(v) for v in residual_signals.values()])
    max_val = np.max(all_vals)
    intensity_threshold = threshold * max_val
    print(f"[DEBUG] Threshold = {intensity_threshold:.2e} ({threshold*100:.1f}% of max {max_val:.2e})")

    
    
    # Obtengo los mzs e intensidades del file original
    for i, spectrum in enumerate(input_map):
        mzs, original_intensities = spectrum.get_peaks()
        mzs = np.array(mzs)
        corrected_intensities = np.zeros_like(mzs)
        #le paso el input_map para poder crear el corrected
        #print(f"[DEBUG] Spectrum {i} has {len(mzs)} peaks.")
        #print(f"[DEBUG] First 5 mzs: {mzs[:5]}")
        #print(f"[DEBUG] First 5 original intensities: {original_intensities[:5]}")
        
        for j, mz in enumerate(mzs):
            idx = np.where(np.abs(mz_keys - mz) <= mz_tol)[0]
            if idx.size > 0:
                closest_mz = mz_keys[idx[np.argmin(np.abs(mz_keys[idx] - mz))]]
                intensity = residual_signals[closest_mz][i]

                if intensity >= intensity_threshold:
                    corrected_intensities[j] = intensity
                    n_corrected += 1
                else:
                    corrected_intensities[j] = 0  # Elimina la oscilación base
            else:
                corrected_intensities[j] = 0  # Fuerza a 0 si no hay corrección posible
        """for j, mz in enumerate(mzs):
            # Busca m/z corregida más cercana dentro de la tolerancia
            idx = np.where(np.abs(mz_keys - mz) <= mz_tol)[0]
            if idx.size > 0:
                closest_mz = mz_keys[idx[np.argmin(np.abs(mz_keys[idx] - mz))]]
                #print(f"[DEBUG] residual_signals[{closest_mz}][{i}] is NaN, skipping correction.")
                corrected_intensities[j] = residual_signals[closest_mz][i]
                n_corrected += 1
                #print(f"[DEBUG] Spectrum {i}, mz {mz:.4f} -> corrected with {closest_mz:.4f}")
                
            else:
                corrected_intensities[j] = original_intensities[j]#si no encuentra esa mz corregida que deje la original
                print(f"[DEBUG] No match for mz={mz:.5f} in spectrum {i}")
            """
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
        #print(f"[DEBUG] Spectrum {i} - First 5 corrected intensities: {corrected_intensities[:5]}")
        corrected_map.addSpectrum(new_spectrum)
        
    print(f"[DEBUG] Total intensities replaced: {n_corrected}")

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
    target_mz_list=[60.07, 65.1, 95.1, 96.085, 110.1173, 922.098]
    #target_mz_list=extract_detected_mzs(rts, mz_array, intensity_array)
    
    auto_mzs=detect_oscillating_mzs(rts, mz_array, intensity_array, target_mz_list, phase_ref, local_freqs_ref)
    
    for mz in auto_mzs:
        if mz not in target_mz_list:
            target_mz_list.append(mz)
        
    cleaned_target_mzs = [float(mz) for mz in target_mz_list]
    #print(f"TARGET MZS DETECTED: {cleaned_target_mzs}")
    
    
    xic_signals = {}
    modulated_signals = {}#Dict[target_mz: float, modulated: np.ndarray]
    residual_signals = {}#Dict[target_mz: float, residual: np.ndarray]

    for target_mz in target_mz_list:
        float(target_mz)
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
    #export_xic_signals_combined_csv(rts, xic_signals, modulated_signals, residual_signals, output_path)
    
    
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

