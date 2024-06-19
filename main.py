import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score,f1_score
import matplotlib.pyplot as plt
import plotly.express as px
import nbformat

def open_pickle(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)

def prediction3(json_path):

    # Importation des données
    RF = open_pickle('RF_best.pkl')

    param = pickle.load(open('dict_pickle.pkl','rb'))


    # Lecture des données
    # data = pd.read_json(json_path)

    # Récupération des données
    arbre = pd.read_csv(json_path)

    # Encodage
    arbre['fk_arb_etat'] = param['label'].transform(arbre['fk_arb_etat'])

    # Sélection des données
 
    x = arbre[['longitude', 'latitude', 'haut_tot', 'tronc_diam', 'haut_tronc', 'age_estim']].copy()
    y = arbre[['fk_arb_etat']].copy()

    # Normalisation
    SC = StandardScaler()
    x_norm = SC.fit_transform(x)

    # Division de la base de données
    x_train, x_test, y_train, y_test = train_test_split(x_norm, y, train_size=0.8, random_state=42)

    # Reshape y_train and y_test
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    #Model
    model_info = {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20, 30]
        }
    }
    best_models = {}
    best_scores = {}
    print("shapes :",pd.DataFrame(x_train).shape,"-",pd.DataFrame(y_train).shape)
    for name, info in model_info.items():
        grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=3, n_jobs=-1, scoring='accuracy')
        grid_search.fit(x_train, y_train)
        best_models[name] = grid_search.best_estimator_
        best_scores[name] = grid_search.best_score_
        print(f"paramètre {name}: {grid_search.best_params_}")
        print(f"model {name} : {grid_search.best_score_}")

    # Prédiction
    RF_pred = RF.predict(x_test)

    print(RF_pred)

    # Calcul des métriques
    RF_precision = precision_score(y_test, RF_pred, average='weighted')
    RF_rappel = recall_score(y_test, RF_pred, average='weighted')
    RF_f1 = f1_score(y_test, RF_pred, average='weighted')

    print(f"Précision: {RF_precision}")
    print(f"Rappel: {RF_rappel}")
    print(f"F1 Score: {RF_f1}")

    print(RF_pred)


    # Affichage sur carte
    y = arbre[['fk_arb_etat', 'latitude', 'longitude']].copy()
    y= pd.concat([y, pd.DataFrame({'prediction': RF_pred})], axis=1)
    y=y.fillna(0)
    print(y)
    fig = px.scatter_mapbox(y, lat= 'latitude', lon='longitude', color='prediction', mapbox_style='open-street-map')
    fig.update_layout(mapbox=dict(style= 'open-street-map', zoom=12, center=dict(lat=y['latitude'].mean(), lon=y['longitude'].mean())), title_text="Arbre potentiel a être arraché")
    fig.show()
        
    # utilisation inverse transforme

prediction3('Data_Arbre.csv')
