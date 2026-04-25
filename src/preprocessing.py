#preprocesamiento de Datos
import pandas as pd

def get_features_and_target(df, target_column='GDS_R3'):
    print(f"\n-Preparando datos")
    print(f"Columna objetivo seleccionada: '{target_column}'")
    
    if target_column not in df.columns:
        raise ValueError(f"La columna '{target_column}' no existe en el DataFrame.")
    
    columnas_gds = [col for col in df.columns if col.startswith('GDS')]
    columnas_excluir = columnas_gds + ['ID']
    
    X_crudo = df.drop(columns=columnas_excluir, errors='ignore')
    y = df[target_column]
    
    if y.isnull().any():
        nulos = y.isnull().sum()
        print(f"Advertencia: Se encontraron {nulos} valores nulos en '{target_column}'. Eliminando esas filas...")
        indices_validos = y.dropna().index
        X_crudo = X_crudo.loc[indices_validos]
        y = y.loc[indices_validos]
        
    print(f"Dimensiones de X (Características): {X_crudo.shape}")
    print(f"Dimensiones de y (Objetivo): {y.shape}")
    
    return X_crudo, y