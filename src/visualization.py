#Visualización
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_gds_distributions_comparative(df, output_dir="../outputs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    columnas_gds = [col for col in df.columns if col.startswith('GDS')]
    
    plt.figure(figsize=(12, 6))
    width = 0.8 / len(columnas_gds)
    
    clases_posibles = np.arange(1, 8)
    
    for i, col in enumerate(columnas_gds):
        conteos = df[col].value_counts().sort_index()
        valores = [conteos.get(c, 0) for c in clases_posibles]
        posiciones = clases_posibles + (i * width) - (0.4) + (width/2)
        plt.bar(posiciones, valores, width=width, label=col, alpha=0.8)
    
    plt.title('Comparación de Frecuencias: Codificaciones GDS', fontsize=14, pad=15)
    plt.xlabel('Nivel de Severidad (Clase)', fontsize=12)
    plt.ylabel('Cantidad de Pacientes', fontsize=12)
    plt.xticks(clases_posibles)
    plt.legend(title="Variables GDS")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    ruta = os.path.join(output_dir, 'distribucion_gds_comparativa.png')
    plt.savefig(ruta)
    plt.close()

def plot_target_class_frequencies(df, target_column, output_dir="../outputs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.figure(figsize=(8, 5))
    
    conteo = df[target_column].value_counts().sort_index()
    
    ax = sns.barplot(x=conteo.index, y=conteo.values, palette="viridis")
    
    plt.title(f'Frecuencia de Clases Objetivo ({target_column})', fontsize=14, pad=15)
    plt.xlabel('Clase (Nivel de Deterioro)', fontsize=12)
    plt.ylabel('Cantidad de Muestras', fontsize=12)
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
        
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    ruta = os.path.join(output_dir, f'frecuencias_objetivo_{target_column}.png')
    plt.savefig(ruta)
    plt.close()


def plot_model_comparison(resultados_modelos, output_dir="../outputs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    nombres_modelos = list(resultados_modelos.keys())
    metricas_nombres = ['Accuracy', 'Precision Macro', 'Recall Macro', 'F1-Score Macro']
    
    datos = {metrica: [] for metrica in metricas_nombres}
    for nombre in nombres_modelos:
        for metrica in metricas_nombres:
            datos[metrica].append(resultados_modelos[nombre][metrica])
            
    x = np.arange(len(nombres_modelos))
    width = 0.2
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for atributo, valor in datos.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, valor, width, label=atributo)
        ax.bar_label(rects, padding=3, fmt='%.2f')
        multiplier += 1
        
    ax.set_ylabel('Puntuación (0.0 - 1.0)', fontsize=12)
    ax.set_title('Comparación de Métricas por Modelo de Ensamble', fontsize=14, pad=20)
    ax.set_xticks(x + width * 1.5, nombres_modelos, rotation=15)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacion_modelos.png'))
    plt.close()

def plot_confusion_matrices(resultados_modelos, output_dir="../outputs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for nombre, metricas in resultados_modelos.items():
        cm = metricas['Matriz de Confusion']
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    linewidths=.5, linecolor='gray')
        
        plt.title(f'Matriz de Confusión\n{nombre}', fontsize=12, pad=10)
        plt.xlabel('Predicción del Modelo', fontsize=10)
        plt.ylabel('Valor Real', fontsize=10)
        
        plt.tight_layout()
        nombre_archivo = nombre.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        plt.savefig(os.path.join(output_dir, f'matriz_confusion_{nombre_archivo}.png'))
        plt.close()

if __name__ == "__main__":
    from data_loader import load_sav_data
    print("probando módulo de visualización")
    try:
        df = load_sav_data("../data/raw/15 atributos R0-R5.sav")
        plot_gds_distributions_comparative(df)
        plot_target_class_frequencies(df, target_column='GDS_R3')
        print("visualizaciones de datos generadas con éxito en la carpeta '../outputs/'")
    except Exception as e:
        print(f"error en la prueba: {e}")