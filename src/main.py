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
from exploracion import evaluar_codificaciones_gds

from visualization import (
    plot_gds_distributions_comparative, 
    plot_model_comparison,
    plot_confusion_matrices,
    plot_target_class_frequencies
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
    print(f"Probabilidades de las 2 primeras muestras de prueba:\n{y_prob}")

if __name__ == "__main__":
    ruta_archivo = "../data/raw/15 atributos R0-R5.sav"
    datos = load_sav_data(ruta_archivo)
    
    mejor_variable = evaluar_codificaciones_gds(datos)
    mejor_modelo = ""
    
    columnas_gds = [col for col in datos.columns if col.startswith('GDS')]
    
    for target in columnas_gds:
        target_output_dir = f"../data/outputs/{target}"
        if not os.path.exists(target_output_dir):
            os.makedirs(target_output_dir)
            
        X, y = get_features_and_target(datos, target_column=target)
        
        if target == mejor_variable:
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

        if target == mejor_variable:
            mejor_f1 = -1
            for nombre_mod, metricas_mod in resultados_totales.items():
                for k, v in metricas_mod.items():
                    if 'F1' in k.upper() and k != 'Matriz de Confusion':
                        if v > mejor_f1:
                            mejor_f1 = v
                            mejor_modelo = nombre_mod
        
        print(f"\nGenerando gráficos para {target} en: {target_output_dir}")
        plot_confusion_matrices(resultados_totales, output_dir=target_output_dir)
        
        try:
            df_resultados = pd.DataFrame(resultados_totales).T
            df_resultados_num = df_resultados.drop(columns=['Matriz de Confusion'])
            pass
        except Exception as e:
            print(f"No se pudo generar la comparacion de modelos para {target}: {e}")

    print(f"\nLa mejor variable fue {mejor_variable} y el mejor modelo fue {mejor_modelo}")