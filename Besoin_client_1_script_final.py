# Importation des librairies
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

nb_clusters = int(input("Veuillez choisir un nombre de catégories : "))
def besoin_1(nb_clusters):
    # Préparation des Données
    # Extraction des données d’intérêt : Sélectionner les colonnes pertinentes de la base de données selon ce besoin.
    data = pd.read_csv("C:/Users/Lenovo/Downloads/Data_Arbre.csv")
    data_selection = data[["longitude", "latitude", "haut_tot"]].copy()
    print(data_selection.head())
    X, y = make_blobs(n_samples=len(data_selection))

    # Apprentissage non supervisé
    # Choix de l'algorithme de clustering : Sélectionner un/des algorithme(s)de clustering pour séparer les arbres en groupes basés sur leur taille.
    # Métriques pour l'apprentissage non supervisé
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_selection[['haut_tot']])

    # Spectral clustering
    spectral = SpectralClustering(n_clusters=nb_clusters)
    labels = spectral.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, labels)
    print("Prédiction 3 : ", labels)
    print("Silhouette score", score)

    # Visualisation sur la carte
    data_selection = pd.concat([data_selection, pd.DataFrame({"cluster": labels})], axis=1)
    fig = px.scatter(data_selection, x="latitude", y="longitude", color="cluster", size="haut_tot")
    fig_1 = px.box(data_selection, x="cluster", y="haut_tot")
    fig_1.update_layout(title_text="Hauteur des arbres dans chaque cluster")
    fig.show()
    fig_1.show()

besoin_1(nb_clusters)


