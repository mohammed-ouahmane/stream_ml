import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, precision_score, recall_score, f1_score, roc_auc_score






# metrics modèles régression

def mse(y_train, y_test):
    # Calcul du MSE
    mse = mean_squared_error(y_train, y_test)
    # Affichage du MSE
    print(f'Erreur quadratique moyenne (MSE) : {mse}')


def r2(y_train, y_test):
    # Calcul du R²
    r2 = r2_score(y_train, y_test)
    # Affichage du R²
    print(f'Coefficient de détermination (R²) : {r2}')


def mae(y_train, y_test):
    # Calcul du MAE
    mae = mean_absolute_error(y_train, y_test)
    # Affichage du MAE
    print(f'Erreur absolue moyenne (MAE) : {mae}')


# metrics modele classification

def precision(y_train, y_test):
    # Calculez la précision
    precision = precision_score(y_train, y_test)
    # Affichez la précision
    print(f'Précision : {precision:.2f}')


def recall(y_train, y_test):
    # Calculez le rappel
    recall = recall_score(y_train, y_test)
    # Affichez le rappel
    print(f'Rappel : {recall:.2f}')


def f1(y_train, y_test):
    # Calculez le F1-score
    f1 = f1_score(y_train, y_test)
    # Affichez le F1-score
    print(f'F1-score : {f1:.2f}')


def auc_roc(y_train, y_prob):
    # Calculez l'AUC-ROC
    auc_roc = roc_auc_score(y_train, y_prob)

    # Affichez l'AUC-ROC
    print(f'AUC-ROC : {auc_roc:.2f}')