# -*- coding: utf-8 -*-

from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt

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

def apply_gaussian_filter(rts, mz_array, intensity_array, target_mz, sigma=50):
    
    """
    Applies gaussian filter correction
    Parameters
    ----------
    target_mz : TYPE
        target mz
    sigma : suavizado del filtro
        DESCRIPTION. The default is 5.

    Returns
    -------
    None.

    """
    xic=build_xic(mz_array, intensity_array, rts, target_mz)
    residual=gaussian_filter1d(xic, sigma)
    
    return xic, residual

def test_best_sigma(rts, mz_array, intensity_array):
    
    sigmas_to_test = [1, 3, 5, 7, 10, 15]

    for sigma in sigmas_to_test:
        xic, filtered = apply_gaussian_filter(rts, mz_array, intensity_array, target_mz=65.1, sigma=sigma)
    
        plt.figure(figsize=(8, 4))
        plt.plot(rts, xic, label="XIC original", alpha=0.5)
        plt.plot(rts, filtered, label=f"Filtrada (σ={sigma})", linewidth=2)
        plt.title(f"Filtro Gaussiano con σ={sigma}")
        plt.xlabel("Tiempo de retención (s)")
        plt.ylabel("Intensidad")
        plt.legend()
        plt.grid(True)
        plt.show()

    
    
    