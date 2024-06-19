import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score,f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
data=pd.read_csv(r"C:\Users\leabo\Desktop\ISEN 3 ème année\projet\Data_Arbre.csv")

data_selected = data[['longitude', 'latitude', 'haut_tot', 'tronc_diam','fk_arb_etat', 'haut_tronc', 'age_estim']].copy()

data_filtered = data_selected[data_selected['fk_arb_etat'].isin(['NON Essouché', 'Essouché'])].copy()

print(data_filtered)

# Encoder la colonne cible (fk_arb_etat)
label_encoder = LabelEncoder()
data_filtered['fk_arb_etat'] = label_encoder.fit_transform(data_filtered['fk_arb_etat'])
print(data_filtered)
# Sauvegarde le modèle d'encodage
#joblib.dump(label_encoder, 'label_encoder.pkl')

# Normalise les données numériques
scaler = StandardScaler()
data_filtered[['haut_tot', 'tronc_diam', 'haut_tronc', 'age_estim']] = scaler.fit_transform(data_filtered[['haut_tot', 'tronc_diam', 'haut_tronc', 'age_estim']])
print(data_filtered)
# Sauvegarde le modèle de normalisation
#joblib.dump(scaler, 'scaler.pkl')

# Sépare les caractéristiques
X = data_filtered.drop('fk_arb_etat', axis=1)
y = data_filtered['fk_arb_etat']

# Divise les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Classes uniques dans y_train :", y_train.unique())
print("Classes uniques dans y_test :", y_test.unique())

if len(y_train.unique()) <= 1 or len(y_test.unique()) <= 1:
    raise ValueError("L'ensemble d'entraînement ou de test contient une seule classe. Assurez-vous que l'ensemble de données est correctement divisé et équilibré.")


# Définir les modèles et les hyperparamètres
models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20, 30]
        }
    },
    'SVC': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params' : {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
}

best_models = {}
best_scores = {}
for name, model_info in tqdm(models.items()):
    grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=2, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    best_scores[name] = grid_search.best_score_
    print(f"Meilleur paramètre {name}: {grid_search.best_params_}")
    print(f"Meilleur model {name} : {grid_search.best_score_}")


best_model = best_models['RandomForest']  
#joblib.dump(best_model, 'best_model.pkl')



y_pred = best_model.predict(X_test)
precision=precision_score(y_test, y_pred)
rappel=recall_score(y_test,y_pred)
f1=f1_score(y_test, y_pred)

print(precision)
print(rappel)
print(f1)

