# Sección: Carga de Datos
import pandas as pd
import os

def load_sav_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"el archivo no se encuentra en la ruta: {filepath}")
    
    print(f"cargando datos desde: {filepath}...")
    df = pd.read_spss(filepath)
    print(f"datos cargados. Dimensiones del archivo: {df.shape}")
    return df

if __name__ == "__main__":
    ruta_archivo = "../data/raw/15 atributos R0-R5.sav"
    try:
        datos = load_sav_data(ruta_archivo)
        print(datos.head())
    except Exception as e:
        print(f"Error al cargar los datos: {e}")