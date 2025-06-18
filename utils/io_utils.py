# utils/io_utils.py
import os
import subprocess

def convert_mzxml_2_mzml(file_path):
    """
    Convierte un archivo .mzXML a .mzML usando ProteoWizard (msconvert).

    Args:
        file_path (str): Ruta del archivo mzXML.

    Returns:
        str: Ruta del nuevo archivo mzML convertido.
    """
    output_folder = os.path.dirname(file_path)
    mzml_file_path = os.path.join(
        output_folder, os.path.basename(file_path).replace(".mzXML", ".mzML")
    )

    try:
        subprocess.run([
            "msconvert", file_path, "--mzML", "--outdir", output_folder,
            "--64", "--zlib", "--filter", "peakPicking true 1-"
        ], check=True)

        if os.path.exists(mzml_file_path):
            return mzml_file_path
        else:
            raise RuntimeError(f"MsConvert no generó el archivo esperado: {mzml_file_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar msconvert: {e}")
        raise RuntimeError("Falló la conversión con ProteoWizard (msconvert).")
