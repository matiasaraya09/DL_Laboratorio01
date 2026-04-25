#Balanceo de datos
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

def balance_training_data(X_train, y_train, random_state=42):
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    return X_resampled, y_resampled