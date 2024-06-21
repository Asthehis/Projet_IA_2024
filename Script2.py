import pandas as pd 
import json
import pickle
import csv

# Récupération des données
arbre = pd.read_csv("Data_Arbre.csv")

# Ecriture du script 
def open_pickle(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)
    
def prediction2(dataFrame):

    # Importation des données
    modeles = open_pickle('dict_pickle2')

    # Lecture des données
    data = pd.read_json(dataFrame)

    # Vérifier qu'il y ait bien les colonnes recquises
    required_columns = ['haut_tronc', 'fk_stadedev', 'tronc_diam', 'haut_tot', 'fk_nomtech']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Vérifier que ce soit encodé, encoder si c'est pas le cas
    OE = modeles['ordinal']
    LE = modeles['label']
    if data['fk_stadedev'].dtype == 'object':
        data[['fk_stadedev']] = OE.transform(data[['fk_stadedev']])
    if data['fk_nomtech'].dtype == 'object':
        data['fk_nomtech'] = LE.transform(data['fk_nomtech'])

    # Sélection des données
    x = data[['haut_tronc', 'fk_stadedev', 'tronc_diam', 'haut_tot', 'fk_nomtech']].copy()
    
    # Vérifier que les données soient normalisées, normaliser sinon
    SC_x = modeles['scaler_x']
    x_norm = SC_x.transform(x)

    # Prédictions
    RF = modeles['RandomForest']
    CART = modeles['DecisionTreeRegressor']
    GBR = modeles['GradientBoostingRegression']
    ETR = modeles['ExtraTreesRegressor']
    RF_pred = RF.predict(x_norm)
    CART_pred = CART.predict(x_norm)
    GBR_pred = GBR.predict(x_norm)
    ETR_pred = ETR.predict(x_norm)


    # Les valeurs prédites seront normalisées, les "dénormaliser"
    SC_y = modeles['scaler_y']
    RF_pred = SC_y.inverse_transform(RF_pred.reshape(-1, 1)).ravel()
    CART_pred = SC_y.inverse_transform(CART_pred.reshape(-1, 1)).ravel()
    GBR_pred =SC_y.inverse_transform(GBR_pred.reshape(-1, 1)).ravel()
    ETR_pred =SC_y.inverse_transform(GBR_pred.reshape(-1, 1)).ravel()

    # Création d'un dictionnaire pour renvoyer les valeurs
    predictions = {
        'RandomForest' : RF_pred.tolist(),
        'DecisionTreeRegressor' : CART_pred.tolist(),
        'GradientBoostingRegression' : GBR_pred.tolist(),
        'ExtraTreesRegressor' : ETR_pred.tolist()
    }

    json_predictions = json.dumps(predictions)

    return json_predictions

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
pred = prediction2('Data_Arbre.json')
print(pred)