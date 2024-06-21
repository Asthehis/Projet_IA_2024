import pandas as pd 
import pickle

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
    y = data[['age_estim']].copy()
    
    # Vérifier que les données soient normalisées, normaliser sinon
    SC_x = modeles['scaler_x']
    SC_y = modeles['scaler_y']
    x_norm = SC_x.transform(x)
    y_norm = SC_y.transform(y)

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
    ETR_pred =SC_y.inverse_transform(ETR_pred.reshape(-1, 1)).ravel()

    # Création d'un dictionnaire pour renvoyer les valeurs
    predictions = {
        'RandomForest' : RF_pred,
        'DecisionTreeRegressor' : CART_pred,
        'GradientBoostingRegression' : GBR_pred,
        'ExtraTreesRegressor' : ETR_pred
    }

    # Création d'un DataFrame pour les prédictions
    predictions_df = pd.DataFrame(predictions)

    # Sauvegarder les prédictions en fichier JSON
    predictions_df.to_json('prediction2.json', orient='records')

    return predictions_df

prediction2('Data_Test2.json')