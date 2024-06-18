import sklearn
import csv
import numpy
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
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

# Utilisation GradientBoostingRegressor
# Détermination des paramètres à tester pour GBR
GBR_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

# Création du modèle
GBR = GradientBoostingRegressor()

# Utilisation de GridSearch pour déterminer les meilleurs paramètres à utiliser pour ce modèle
GBR_GS = GridSearchCV(estimator=GBR, param_grid=GBR_param_grid, cv=3, n_jobs=-1, verbose=2)

# Entraînement GridSearch
GBR_GS.fit(x_train, y_train)

# Récupération et affichage des meilleurs paramètres
GBR_best_params = GBR_GS.best_params_
print("GBR Best parameters found: ", GBR_best_params)

# Récupération du meilleur estimateur et prédiction
best_GBR = GBR_GS.best_estimator_
GBR_y_pred = best_GBR.predict(x_test)

# Utilisation SVR
# Détermination des paramètres à tester pour SVR
SVR_param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
"""
# Création du modèle
SVR_model = SVR()

# Utilisation de GridSearch pour déterminer les meilleurs paramètres à utiliser pour ce modèle
SVR_GS = GridSearchCV(estimator=SVR_model, param_grid=SVR_param_grid, cv=3, n_jobs=-1, verbose=2)

# Entraînement GridSearch
SVR_GS.fit(x_train, y_train)

# Récupération et affichage des meilleurs paramètres
SVR_best_params = SVR_GS.best_params_
print("SVR Best parameters found: ", SVR_best_params)

# Récupération du meilleur estimateur et prédiction
best_SVR = SVR_GS.best_estimator_
SVR_y_pred = best_SVR.predict(x_test)
"""
# Évaluation des modèles et affichages des scores obtenus
RF_mse = mean_squared_error(y_test, RF_y_pred)
RF_r2 = r2_score(y_test, RF_y_pred)
CART_mse = mean_squared_error(y_test, CART_y_pred)
CART_r2 = r2_score(y_test, CART_y_pred)
GBR_mse = mean_squared_error(y_test, GBR_y_pred)
GBR_r2 = r2_score(y_test, GBR_y_pred)
#SVR_mse = mean_squared_error(y_test, SVR_y_pred)
#SVR_r2 = r2_score(y_test, SVR_y_pred)
print(f"Random Forest -> Mean Squared Error: {RF_mse}, R^2 Score: {RF_r2}")
print(f"DecisionTreeRegression -> Mean Squared Error: {CART_mse}, R^2 Score: {CART_r2}")
print(f"GradientBoostingRegressor -> Mean Squared Error: {GBR_mse}, R^2 Score: {GBR_r2}")
#print(f"SupportVectorRegressor -> Mean Squared Error: {SVR_mse}, R^2 Score: {SVR_r2}")
