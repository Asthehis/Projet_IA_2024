import sklearn
import csv
import numpy
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Récupération des données
arbre = pd.read_csv("Data_Arbre.csv")

# Méthode regression Random Forest, CART...Gini index
# Sélection des données intéressantes pour l'étude
x = arbre[['haut_tronc', 'fk_prec_estim', 'tronc_diam', 'haut_tot', 'fk_stadedev']].copy()
y = arbre[['age_estim']].copy()

# Encodage
encoder = OrdinalEncoder()
x['fk_stadedev'] = encoder.fit_transform(x[['fk_stadedev']])

# Normalisation
x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (y - y.min()) / (y.max() - y.min())

# Division de la base de données
x_train, x_test, y_train, y_test = train_test_split(x_norm, y_norm, train_size=0.8, random_state=42)

# Reshape y_train and y_test
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Utilisation Random Forest
# Détermination des paramètres à tester pour le RandomForest
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Création du modèle
RF = RandomForestRegressor()

# Utilisation de GridSearch pour déterminer les meilleurs paramètres à utiliser pour ce modèle
RF_GS = GridSearchCV(estimator=RF, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Entraînement du GridSearch
RF_GS.fit(x_train, y_train)

# Récupération et affichage des meilleurs paramètres
best_params = RF_GS.best_params_
print("Best parameters found: ", best_params)

# Récupération du meilleur estimateur et prédiction
best_RF = RF_GS.best_estimator_
y_pred = best_RF.predict(x_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"n_estimators=1 -> Mean Squared Error: {mse}, R^2 Score: {r2}")

# Tests avec plusieurs n_estimateurs (nombre d'arbres) 10, 50, 100 et 200
"""
for n in [10, 50, 100, 200]:
    RF = RandomForestRegressor(n_estimators=n)
    RF.fit(x_train, y_train)
    y_pred = RF.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"n_estimators={n} -> Mean Squared Error: {mse}, R^2 Score: {r2}")
"""

# Utilisation CART
DTR = DecisionTreeRegressor(random_state=0, criterion="poisson").fit(x_train, y_train)

# Prédiction
y_pred = DTR.predict(x_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"DecisionTreeRegression -> Mean Squared Error: {mse}, R^2 Score: {r2}")
