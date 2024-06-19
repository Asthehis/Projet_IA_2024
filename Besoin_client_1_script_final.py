# Importation des librairies
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import DBSCAN


nb_clusters = int(input("Veuillez choisir un nombre de catégories : "))
def besoin_1(nb_clusters):
    # Préparation des Données
    # Extraction des données d’intérêt : Sélectionner les colonnes pertinentes de la base de données selon ce besoin.
    data = pd.read_csv("Data_Arbre.csv")
    data_selection = data[["longitude", "latitude", "haut_tot"]].copy()
    print(data_selection.head())

    # Apprentissage non supervisé
    # Choix de l'algorithme de clustering : Sélectionner un/des algorithme(s)de clustering pour séparer les arbres en groupes basés sur leur taille.
    # Métriques pour l'apprentissage non supervisé
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_selection[['haut_tot']])

    # K-Means
    k_means = KMeans(n_clusters=nb_clusters)
    labels = k_means.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, labels)
    print("Prédiction : ", labels)
    print("Silhouette score", score)


    # Visualisation sur la carte
    data_selection = pd.concat([data_selection, pd.DataFrame({"cluster": labels})], axis=1)
    fig = px.scatter_mapbox(data_selection, lat="latitude", lon="longitude", color="cluster",  mapbox_style='open-street-map')
    fig_1 = px.box(data_selection, x="cluster", y="haut_tot")
    fig.update_layout(mapbox=dict(style="open-street-map",zoom=12,center=dict(lat=data_selection["latitude"].mean(), 
    lon=data_selection["longitude"].mean())),title_text="Dispersion des arbres par cluster")
    fig_1.update_layout(title_text="Hauteur des arbres dans chaque cluster")
    fig.show()
    fig_1.show()

    # Fonctionnalité supplémentaire : Détection des anomalies
    # tronc_diam, fk_prec_estim
    data_anomalies = data[["longitude", "latitude", "fk_prec_estim", "tronc_diam"]].copy()
    data_anomalies_scaled = scaler.fit_transform(data_anomalies)
    dbscan = DBSCAN(eps=1.5, min_samples=8) # 2 à 4 fois le nombre de colonnes choisi
    clusters = dbscan.fit_predict(data_anomalies_scaled)
    data_anomalies['cluster'] = clusters
    outliers = data_anomalies[data_anomalies['cluster'] == -1]
    print("Number of outliers :", len(outliers))
    plt.figure(figsize=(10,13))
    plt.scatter(data_anomalies['fk_prec_estim'], data_anomalies['tronc_diam'], c=data_anomalies['cluster'], cmap='coolwarm', label = 'Clusters')
    plt.scatter(outliers['fk_prec_estim'], outliers['tronc_diam'], c='black', label='Outliers', marker='x')
    plt.xlabel("Précision de l'âge estimé")
    plt.ylabel("Diamètre du tronc")
    plt.title("Détection des Anomalies des Arbres avec DBSCAN, en fonction de la précision de l'âge estimé et du diamètre du tronc")
    plt.show()
    # haut_tot, diam_tronc
    data_anomalies = data[["longitude", "latitude", "haut_tot", "tronc_diam"]].copy()
    data_anomalies_scaled = scaler.fit_transform(data_anomalies)
    dbscan = DBSCAN(eps=1.5, min_samples=8) # 2 à 4 fois le nombre de colonnes choisi
    clusters = dbscan.fit_predict(data_anomalies_scaled)
    data_anomalies['cluster'] = clusters
    outliers = data_anomalies[data_anomalies['cluster'] == -1]
    print("Number of outliers :", len(outliers))
    plt.figure(figsize=(10,13))
    plt.scatter(data_anomalies['haut_tot'], data_anomalies['tronc_diam'], c=data_anomalies['cluster'], cmap='coolwarm', label = 'Clusters')
    plt.scatter(outliers['haut_tot'], outliers['tronc_diam'], c='black', label='Outliers', marker='x')
    plt.xlabel("Hauteur totale")
    plt.ylabel("Diamètre du tronc")
    plt.title("Détection des Anomalies des Arbres avec DBSCAN, en fonction de la hauteur totale et du diamètre du tronc")
    plt.show()
    # age_estim, tronc_diam
    data_anomalies = data[["longitude", "latitude", "age_estim", "tronc_diam"]].copy()
    data_anomalies_scaled = scaler.fit_transform(data_anomalies)
    dbscan = DBSCAN(eps=1.5, min_samples=8) # 2 à 4 fois le nombre de colonnes choisi
    clusters = dbscan.fit_predict(data_anomalies_scaled)
    data_anomalies['cluster'] = clusters
    outliers = data_anomalies[data_anomalies['cluster'] == -1]
    print("Number of outliers :", len(outliers))
    plt.figure(figsize=(10,13))
    plt.scatter(data_anomalies['age_estim'], data_anomalies['tronc_diam'], c=data_anomalies['cluster'], cmap='coolwarm', label = 'Clusters')
    plt.scatter(outliers['age_estim'], outliers['tronc_diam'], c='black', label='Outliers', marker='x')
    plt.xlabel("Age estimé")
    plt.ylabel("Diamètre du tronc")
    plt.title("Détection des Anomalies des Arbres avec DBSCAN, en fonction de l'âge estimé et du diamètre du tronc")
    plt.show()



besoin_1(nb_clusters)


