#Evaluación de Modelos
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from balancing import balance_training_data

def evaluate_model(model, X, y, n_splits=5, apply_balancing=True, k_mejores=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    matriz_acumulada = None 
    
    X_arr = X.values if hasattr(X, 'values') else X
    y_arr = y.values if hasattr(y, 'values') else y
    
    for train_index, test_index in skf.split(X_arr, y_arr):
        X_train, X_test = X_arr[train_index], X_arr[test_index]
        y_train, y_test = y_arr[train_index], y_arr[test_index]
        
        selector = SelectKBest(score_func=chi2, k=k_mejores)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        
        if apply_balancing:
            X_train, y_train = balance_training_data(X_train, y_train)
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred)
        if matriz_acumulada is None:
            matriz_acumulada = cm
        else:
            matriz_acumulada += cm
            
    return {
        'Accuracy': np.mean(accuracies),
        'Precision Macro': np.mean(precisions),
        'Recall Macro': np.mean(recalls),
        'F1-Score Macro': np.mean(f1_scores),
        'Matriz de Confusion': matriz_acumulada
    }