#Boosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def get_boosting_model():
    modelo_base = DecisionTreeClassifier(max_depth=1, random_state=42)
    modelo_boosting = AdaBoostClassifier(
        estimator=modelo_base,
        n_estimators=100,
        learning_rate=0.5,
        algorithm='SAMME',
        random_state=42
    )
    return modelo_boosting