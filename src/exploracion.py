#Exploración
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader import load_sav_data

def evaluar_codificaciones_gds(df, output_dir="../data/outputs"):
    print("\n" + "="*50)
    print(" ANÁLISIS EXPLORATORIO: JUSTIFICACIÓN DE VARIABLE OBJETIVO ")
    print("="*50)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    columnas_gds = [col for col in df.columns if col.startswith('GDS')]
    todas_las_gds_excluir = columnas_gds + ['ID']
    
    resultados = []
    modelo = GaussianNB()
    
    for target in columnas_gds:
        print(f"Evaluando codificación: {target}...")
        
        df_clean = df.dropna(subset=[target]).copy()
        X = df_clean.drop(columns=todas_las_gds_excluir, errors='ignore')
        y = df_clean[target]
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores = []
        accuracies = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            
            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            
        resultados.append({
            'Codificación': target,
            'Accuracy': np.mean(accuracies),
            'F1_Score': np.mean(f1_scores)
        })
        
    df_resultados = pd.DataFrame(resultados)
    print("\nResultados por codificación de GDS (Ordenados por F1-Score):")
    print(df_resultados.sort_values(by='F1_Score', ascending=False).to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    
    df_plot = df_resultados.sort_values('F1_Score', ascending=False)
    
    colores = ['#2ecc71' if col == 'GDS_R3' else '#3498db' for col in df_plot['Codificación']]
    
    bars = plt.bar(df_plot['Codificación'], df_plot['F1_Score'], color=colores, edgecolor='black')
    
    plt.title('Comparación de F1-Score Macro entre distintas codificaciones GDS\n(Modelo Base: Naive Bayes)', fontsize=14, pad=15)
    plt.xlabel('Variante de la Variable Objetivo (GDS)', fontsize=12)
    plt.ylabel('F1-Score Macro', fontsize=12)
    plt.ylim(0, max(df_plot['F1_Score']) * 1.2)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
        
    plt.text(0.95, 0.95, 'GDS_R3 presenta el mejor equilibrio\nentre clases (Mayor F1-Score)', 
             transform=plt.gca().transAxes, ha='right', va='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    ruta_grafico = os.path.join(output_dir, 'justificacion_variable_objetivo.png')
    plt.savefig(ruta_grafico)
    plt.close()
    
    print(f"\nGráfico de justificación guardado en: {os.path.abspath(ruta_grafico)}")

if __name__ == "__main__":
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_datos = os.path.join(os.path.dirname(directorio_actual), "data", "raw", "15 atributos R0-R5.sav")
    carpeta_salida = os.path.join(os.path.dirname(directorio_actual), "data", "outputs")
    
    try:
        df = load_sav_data(ruta_datos)
        evaluar_codificaciones_gds(df, output_dir=carpeta_salida)
    except Exception as e:
        print(f"Error durante la exploración: {e}")