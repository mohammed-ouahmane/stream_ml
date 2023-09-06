import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import plotly.express as px
import scipy.stats as stats
import seaborn as sns
import mpld3
import streamlit.components.v1 as components



def digramme_dispersion(y_train, y_pred):

    plt.scatter(y_train, y_pred, color='blue', marker='o', label='Données réelles vs. Prédites')
    # Ajoutez une ligne de référence (y=x) pour montrer une prédiction parfaite
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', lw=2, label='Prédiction parfaite')
    # Ajoutez des étiquettes et une légende
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Valeurs prédites')
    plt.legend(loc='best')
    # Affichez le diagramme de dispersion
    plt.title('Diagramme de dispersion entre les valeurs réelles et prédites')
    plt.grid(True)
    plt.show()


def courbe_regression(X, y, y_pred):

    # Créez un graphique avec les points de données réelles
    plt.scatter(X, y, color='blue', label='Données réelles')
    # Tracez la ligne de régression (ligne de prédiction) en utilisant les valeurs prédites
    plt.plot(X, y_pred, color='red', linestyle='-', linewidth=2, label='Ligne de régression')
    plt.xlabel('Caractéristiques')
    plt.ylabel('Valeurs réelles / prédites')
    plt.legend(loc='best')
    plt.title('Courbe de régression entre les données réelles et prédites')
    plt.grid(True)
    plt.show()


def histo_residu(y_test, y_pred):
    # Calculez les résidus en soustrayant les valeurs réelles des valeurs prédites
    residus = y_test - y_pred

    # Créez un histogramme Plotly
    fig = px.histogram(residus, nbins=10)
    fig.update_traces(marker=dict(color='blue', opacity=0.7, line=dict(color='black', width=1)))
    fig.update_layout(xaxis_title='Résidus', yaxis_title='Fréquence', title='Histogramme des résidus')
    
    return fig

    # Créez une fonction pour tracer la courbe d'apprentissage
def courbe_appr(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 50))

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.title('Courbe d\'apprentissage')
    plt.xlabel('Taille de l\'ensemble d\'entraînement')
    plt.ylabel('Erreur quadratique moyenne')

    plt.plot(train_sizes, train_scores_mean, label='Score d\'entraînement', color='blue', marker='o')
    plt.plot(train_sizes, test_scores_mean, label='Score de validation', color='red', marker='o')

    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def quant_quant(y_test, y_pred):
    
    # Supposons que "residus" contienne les résidus de votre modèle.
    residus = y_test - y_pred

    # Calculez les quantiles des résidus
    sorted_residus = np.sort(residus)
    normal_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residus)))

    # Créez un graphique QQ
    plt.figure(figsize=(6, 6))
    plt.scatter(normal_quantiles, sorted_residus, color='blue')
    plt.plot([min(normal_quantiles), max(normal_quantiles)], [min(normal_quantiles), max(normal_quantiles)], color='red', linestyle='--')
    plt.xlabel('Quantiles théoriques (distribution normale)')
    plt.ylabel('Quantiles observés (résidus)')
    plt.title('Graphique QQ')
    plt.grid(True)
    plt.show()


def conf_matrix(y_train, y_pred):

    # Calculez la matrice de confusion
    conf_matrix = confusion_matrix(y_train, y_pred)

    # Créez un graphique de la matrice de confusion
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.xlabel('Valeurs Prédites')
    plt.ylabel('Valeurs Réelles')
    plt.title('Matrice de Confusion')
    plt.show()


def roc_class(X_train, X_test, y_train, y_test):

    clf_tree = DecisionTreeClassifier()
    clf_tree.fit(X_train, y_train)

    y_score1 = clf_tree.predict_proba(X_test)[:,1]

    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)

    plt.subplots(1, figsize=(10, 10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate1, true_positive_rate1, label='Courbe ROC')
    plt.plot([0, 1], ls="--", label='Ligne de Référence')
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # Ajoutez la légende
    plt.legend(loc='lower right')

    plt.show()


def disp_classes(X, y): # ne fonctionne pas, WIP

    # Réduisez les dimensions à 2D en utilisant l'analyse en composantes principales (PCA)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    y = y.to_numpy()

    # Créez un diagramme de dispersion des classes
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red']
    for i in range(2):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], color=colors[i], label=f'Classe {i}')
    plt.xlabel('Composante Principale 1')
    plt.ylabel('Composante Principale 2')
    plt.legend()
    plt.title('Diagramme de Dispersion des Classes (2D)')
    plt.show()

def courbe_prec_recall(y_true, y_scores):
    # Calculer la courbe de précision-rappel
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # Calculer l'aire sous la courbe de précision-rappel (AUC-PR)
    auc_pr = average_precision_score(y_true, y_scores)

    # Tracez la courbe de précision-rappel
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label='Courbe PR (AUC = %0.2f)' % auc_pr)
    plt.xlabel('Rappel (Recall)')
    plt.ylabel('Précision (Precision)')
    plt.title('Courbe de Précision-Rappel')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def distribution_target(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Distribution de la variable cible')
    plt.show()

def etude_correlation(data):
    data1 =data
    data1['target']=data1['target'].astype('category').cat.codes

    mask = np.triu(data1.corr())
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = sns.diverging_palette(15, 160, n=11, s=100)
    sns.heatmap(data1.corr(),
            mask=mask,
            annot=True,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax)