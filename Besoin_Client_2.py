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
RF_param_grid = {
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
RF_GS = GridSearchCV(estimator=RF, param_grid=RF_param_grid, cv=3, n_jobs=-1, verbose=2)

# Entraînement du GridSearch
RF_GS.fit(x_train, y_train)

# Récupération et affichage des meilleurs paramètres
RF_best_params = RF_GS.best_params_
print("RF Best parameters found: ", RF_best_params)

# Récupération du meilleur estimateur et prédiction
best_RF = RF_GS.best_estimator_
RF_y_pred = best_RF.predict(x_test)


# Utilisation CART
# Détermination des paramètres à tester pour CART
CART_param_grid = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20, 30, 40],
    'min_impurity_decrease': [0.0, 0.01, 0.1]
}

# Création du modèle
DTR = DecisionTreeRegressor()

# Utilisation de GridSearch pour déterminer les meilleurs paramètres à utiliser pour ce modèle
CART_GS = GridSearchCV(estimator=DTR, param_grid=CART_param_grid, cv=3, n_jobs=-1, verbose=2)

# Entraînement GridSearch
CART_GS.fit(x_train, y_train)

# Récupération et affichage des meilleurs paramètres
CART_best_params = CART_GS.best_params_
print("CART Best parameters found: ", CART_best_params)

# Récupération du meilleur estimateur et prédiction
best_CART = CART_GS.best_estimator_
CART_y_pred = best_CART.predict(x_test)

# Évaluation du modèle
RF_mse = mean_squared_error(y_test, RF_y_pred)
RF_r2 = r2_score(y_test, RF_y_pred)
CART_mse = mean_squared_error(y_test, CART_y_pred)
CART_r2 = r2_score(y_test, CART_y_pred)
print(f"Random Forest -> Mean Squared Error: {RF_mse}, R^2 Score: {RF_r2}")
print(f"DecisionTreeRegression -> Mean Squared Error: {CART_mse}, R^2 Score: {CART_r2}")
