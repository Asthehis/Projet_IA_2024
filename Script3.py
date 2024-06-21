import pandas as pd 
import json
import pickle
import csv
import plotly.express as px

# Récupération des données
arbre = pd.read_csv("Data_Arbre.csv")

def open_pickle(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)

def prediction3(dataFrame):

    # Importation des données
    modeles = open_pickle('dict_pickle3.pkl')

    # Lecture des données
    data = pd.read_json(dataFrame)

    # Vérifier qu'il y ait bien les données
    required_columns = ['longitude', 'latitude', 'haut_tot', 'tronc_diam', 'haut_tronc', 'age_estim']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
        
    # Vérifier que ce soit encodé
    LE = modeles['label']
    if data['fk_arb_etat'].dtype == 'object':
        data['fk_arb_etat'] = LE.transform(data['fk_arb_etat'])

    # Filtrer les données pour n'avoir que les essouché et non essouché
    filtered_data = data[data['fk_arb_etat'].isin(LE.transform(['Non essouché', 'Essouché']))]

    # Sélection des données
    x = filtered_data[['longitude', 'latitude', 'haut_tot', 'tronc_diam', 'haut_tronc', 'age_estim']].copy()

    # Normalisation
    SC = modeles['scaler']
    x_norm = SC.transform(x)

    # Prédiction
    RF = modeles['RandomForest']
    GBC = modeles['GradientBoostingClassifier']
    RF_pred = RF.predict(x_norm)
    GBC_pred = GBC.predict(x_norm)

    # Affichage sur carte
    y = data[['fk_arb_etat', 'latitude', 'longitude']].copy()
    y= pd.concat([y, pd.DataFrame({'prediction': RF_pred})], axis=1)
    y=y.fillna(0)
    print(y)
    fig = px.scatter_mapbox(y, lat= 'latitude', lon='longitude', color='prediction', mapbox_style='open-street-map')
    fig.update_layout (mapbox=dict(style= 'open-street-map', zoom=12, center=dict(lat=y['latitude'].mean(), lon=y['longitude'].mean())), title_text="Arbre potentiel a être arraché")
    fig.show()

def csv_to_json(csvFilePath, jsonFilePath):
    jsonArray = []
      
    #read csv file
    with open(csvFilePath, encoding='utf-8') as csvf: 
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf) 

        #convert each csv row into python dict
        for row in csvReader: 
            #add this python dict to json array
            jsonArray.append(row)
  
    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)

csv_to_json('Data_Arbre.csv', 'Data_Arbre.json')
prediction3('Data_Arbre.json')