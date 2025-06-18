# -*- coding: utf-8 -*-
# main.py
# -*- coding: utf-8 -*-

from processing.processor2 import process_file

import os

if __name__ == "__main__":
    

    file_path = "C:/Users/maipa/repos/oscilations_TFG_repository/FILES/TESTS/MSCONVERT_CTRL_103_01_c_afterreboot_original.mzML"
    save_as = "C:/Users/maipa/repos/oscilations_TFG_repository/FILES/TESTS/CTRL_103_01_c_afterreboot_corrected.mzML"

    if os.path.exists(save_as):
        print(f"El archivo {save_as} ya existe. Eliminándolo para reemplazarlo...")
        os.remove(save_as)

    process_file(file_path, save_as)


