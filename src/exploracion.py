import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            
        resultados.append({'Codificacion': target, 'F1_Score': np.mean(f1_scores)})
        
    df_plot = pd.DataFrame(resultados)
    mejor_cod = df_plot.loc[df_plot['F1_Score'].idxmax(), 'Codificacion']
    
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71' if col == mejor_cod else '#3498db' for col in df_plot['Codificacion']]
    bars = plt.bar(df_plot['Codificacion'], df_plot['F1_Score'], color=colors, edgecolor='black')
    
    plt.title('Comparación de F1-Score Macro entre distintas codificaciones GDS\n(Modelo Base: Naive Bayes)', fontsize=14, pad=15)
    plt.xlabel('Variante de la Variable Objetivo (GDS)', fontsize=12)
    plt.ylabel('F1-Score Macro', fontsize=12)
    plt.ylim(0, max(df_plot['F1_Score']) * 1.2)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
        
    plt.text(0.95, 0.95, f'{mejor_cod} presenta el mejor equilibrio\nentre clases (Mayor F1-Score)', 
             transform=plt.gca().transAxes, ha='right', va='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    ruta_grafico = os.path.join(output_dir, 'justificacion_variable_objetivo.png')
    plt.savefig(ruta_grafico)
    plt.close()
    
    print(f"\nGráfico de justificación guardado en: {os.path.abspath(ruta_grafico)}")
    
    return mejor_cod