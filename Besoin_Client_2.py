import sklearn
import csv
import numpy
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Récupération des données
arbre = pd.read_csv("Data_Arbre.csv")
#print(arbre)

# Méthode regression Random Forest, CART...Gini index
# Sélection des données intéressantes pour l'étude
x = arbre[['haut_tronc', 'fk_prec_estim', 'tronc_diam', 'haut_tot', 'fk_stadedev']].copy()
y = arbre[['age_estim']].copy()
print(x)
print(y)

# Encodage
encoder = OrdinalEncoder()
x['fk_stadedev'] = encoder.fit_transform(x[['fk_stadedev']])
print(x['fk_stadedev'].head())

# Normalisation
x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (y - y.min()) / (y.max() - y.min())

# Division de la base de données
x_train, x_test, y_train, y_test = train_test_split(x_norm, y_norm, train_size=0.8, random_state=42)

# Reshape y_train and y_test
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Création et entraînement du modèle
RF = RandomForestRegressor(n_estimators=1)
RF = RF.fit(x_train, y_train)

# Prédiction
y_pred = RF.predict(x_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"n_estimators=1 -> Mean Squared Error: {mse}, R^2 Score: {r2}")

# Tests avec plusieurs n_estimateurs (nombre d'arbre) 10, 50, 100 et 200
for n in [10, 50, 100, 200]:
    RF = RandomForestRegressor(n_estimators=n)
    RF.fit(x_train, y_train)
    y_pred = RF.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"n_estimators={n} -> Mean Squared Error: {mse}, R^2 Score: {r2}")