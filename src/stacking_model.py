#Stacking
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def get_stacking_model():
    modelos_base = [
        ('nb', GaussianNB()),
        ('tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('lr', LogisticRegression(max_iter=2000, random_state=42))
    ]
    meta_modelo = LogisticRegression(max_iter=2000, random_state=42)
    
    modelo_stacking = StackingClassifier(
        estimators=modelos_base,
        final_estimator=meta_modelo,
        cv=5,
        n_jobs=-1
    )
    return modelo_stacking