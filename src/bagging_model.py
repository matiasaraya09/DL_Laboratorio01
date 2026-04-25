#Modelo Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def get_bagging_model():
    modelo_base = DecisionTreeClassifier(random_state=42)
    modelo_bagging = BaggingClassifier(
        estimator=modelo_base,
        n_estimators=50,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    return modelo_bagging