#main.py
import os
import warnings
import pandas as pd
import matplotlib

matplotlib.use('Agg') 

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from data_loader import load_sav_data
from preprocessing import get_features_and_target
from bagging_model import get_bagging_model
from boosting_model import get_boosting_model
from stacking_model import get_stacking_model
from evaluation import evaluate_model

from visualization import (
    plot_gds_distributions_comparative, 
    plot_model_comparison,
    plot_confusion_matrices
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def demostracion_probabilidades_bayes(X, y):
    print("\n" + "-"*50)
    print(" PROBABILIDADES (NAIVE BAYES) ")
    print("-"*50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    
    y_prob = nb.predict_proba(X_test[:2])
    y_pred = nb.predict(X_test[:2])
    
    y_test_vals = y_test.values if hasattr(y_test, 'values') else y_test
    
    for i in range(2):
        print(f"Paciente {i+1}:")
        probs_formateadas = [round(p, 4) for p in y_prob[i]]
        print(f"  - Probabilidades por clase: {probs_formateadas}")
        print(f"  - Clase predicha por Bayes: {y_pred[i]}")
        print(f"  - Clase real del paciente: {y_test_vals[i]}")

if __name__ == "__main__":
    ruta_datos = "../data/raw/15 atributos R0-R5.sav"
    carpeta_salida = "../data/outputs"
    
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        
    df_raw = load_sav_data(ruta_datos)
    
    plot_gds_distributions_comparative(df_raw, output_dir=carpeta_salida)
    
    columnas_gds = [col for col in df_raw.columns if col.startswith('GDS')]
    
    for target in columnas_gds:
        print("\n" + "-"*50)
        print(f"Iniciando experimento y codificación de la siguiente variable: {target} ")
        print("-"*50)
        
        target_output_dir = os.path.join(carpeta_salida, target)
        if not os.path.exists(target_output_dir):
            os.makedirs(target_output_dir)
            
        X, y = get_features_and_target(df_raw, target_column=target)
        
        if target == columnas_gds[0]:
            demostracion_probabilidades_bayes(X, y)
        
        modelos = {
            "Naive Bayes (Baseline)": GaussianNB(),
            "Bagging (Trees)": get_bagging_model(),
            "AdaBoost (Stumps)": get_boosting_model(),
            "Stacking (NB+Tree+LR)": get_stacking_model()
        }
        
        print("\n" + "-"*50)
        print(f" EVALUACIÓN DE LOS MODELOS ({target}) ")
        print("-"*50)
        
        resultados_totales = {}
        
        for nombre, modelo in modelos.items():
            print(f"\nEntrenando y evaluando el modelo: {nombre}")
            
            metricas = evaluate_model(modelo, X, y, apply_balancing=True)
            resultados_totales[nombre] = metricas 
            
            for metrica, valor in metricas.items():
                if metrica == 'Matriz de Confusion':
                    print(f"  - {metrica} generada.")
                else:
                    print(f"  - {metrica}: {valor:.4f}")
        
        print(f"\nGenerando gráficos para {target} en: {target_output_dir}")
        plot_confusion_matrices(resultados_totales, output_dir=target_output_dir)
        
        try:
            df_resultados = pd.DataFrame(resultados_totales).T
            df_resultados_num = df_resultados.drop(columns=['Matriz de Confusion'])
            plot_model_comparison(df_resultados_num, output_dir=target_output_dir)
        except Exception as e:
            pass 
            
    print("\n" + "-"*50)
    print("Fin del Lab01")
    print("-"*50)