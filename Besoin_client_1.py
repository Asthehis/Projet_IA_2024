# Importation des librairies
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# Préparation des Données
# Extraction des données d’intérêt : Sélectionner les colonnes pertinentes de la base de données selon ce besoin.
data = pd.read_csv("C:/Users/Lenovo/Downloads/Data_Arbre.csv")
data_selection = data[["longitude", "latitude", "haut_tot"]].copy()
print(data_selection.head())
X, y = make_blobs(n_samples=len(data_selection))


# Apprentissage non supervisé
# Choix de l'algorithme de clustering : Sélectionner un/des algorithme(s)de clustering pour séparer les arbres en groupes basés sur leur taille.
# Métriques pour l'apprentissage non supervisé
# K-Means
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selection[['haut_tot']])


# silhouette_scores = []
# for k in range(2, 10):
#     kmeans = KMeans(n_clusters=k, random_state=0)
#     labels = kmeans.fit_predict(data_scaled)
#     score = silhouette_score(data_scaled, labels)
#     silhouette_scores.append(score)
#     print("Silhouette score", score)
#     print("Nombre cluster", k)
#     print("Prédiction  : ", labels)
#     NMI = normalized_mutual_info_score(y, labels)
#     print("NMI", NMI)
# plt.plot(range(2, 10), silhouette_scores, marker='o')
# plt.xlabel('Nombre de clusters')
# plt.ylabel('Score de silhouette')
# plt.title('Score de silhouette pour différents nombres de clusters')
# plt.show()

# # Mean shift
# bandwidth = estimate_bandwidth(X)
# print("Bandwidth", bandwidth)
# shift = MeanShift(bandwidth=bandwidth)
# predict_2 = shift.fit_predict(data_selection[["haut_tot"]])
# print("Prédiction 2 : ", predict_2)
# NMI_2 = normalized_mutual_info_score(y, predict_2)
# print("NMI 2", NMI_2)

# Spectral clustering
silhouette_scores = []
for k in range(2, 10):
    spectral = SpectralClustering(n_clusters=3)
    labels_2 = spectral.fit_predict(data_scaled)
    score_2 = silhouette_score(data_scaled, labels_2)
    silhouette_scores.append(score_2)
    print("Silhouette score", score_2)
    print("Nombre cluster", k)
    print("Prédiction 3 : ", labels_2)
    NMI_3 = normalized_mutual_info_score(y, labels_2)
    print("NMI 3",NMI_3)
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de silhouette')
plt.title('Score de silhouette pour différents nombres de clusters')
plt.show()

# # Agglomerative clustering
# silhouette_scores = []
# for k in range(2, 10):
#     ward = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage = 'ward')
#     labels_3 = ward.fit_predict(data_scaled)
#     score_3 = silhouette_score(data_scaled, labels_3)
#     silhouette_scores.append(score_3)
#     print("Silhouette score", score_3)
#     print("Nombre cluster", k)
#     print("Prédiction 4 : ", labels_3)
#     NMI_4 = normalized_mutual_info_score(y, labels_3)
#     print("NMI 4",NMI_4)
# plt.plot(range(2, 10), silhouette_scores, marker='o')
# plt.xlabel('Nombre de clusters')
# plt.ylabel('Score de silhouette')
# plt.title('Score de silhouette pour différents nombres de clusters')
# plt.show()

# # Birch
# silhouette_scores = []
# for k in range(2, 10):
#     birch = Birch(n_clusters=k)
#     labels_4 = birch.fit_predict(data_scaled)
#     score_4 = silhouette_score(data_scaled, labels_4)
#     silhouette_scores.append(score_4)
#     print("Silhouette score", score_4)
#     print("Nombre cluster", k)
#     print("Prédiction 5 : ", labels_4)
#     NMI_5 = normalized_mutual_info_score(y, labels_4)
#     print("NMI 5",NMI_5)
# plt.plot(range(2, 10), silhouette_scores, marker='o')
# plt.xlabel('Nombre de clusters')
# plt.ylabel('Score de silhouette')
# plt.title('Score de silhouette pour différents nombres de clusters')
# plt.show()


# Visualisation sur la carte
cluster = []
spectral_test = SpectralClustering(n_clusters=3)
labels_test = spectral_test.fit_predict(data_scaled)
print(labels_test)
data_selection = pd.concat([data_selection,pd.DataFrame({"cluster":labels_test})],axis=1)
print(data_selection)
fig = px.scatter(data_selection, x = "latitude", y ="longitude", color = "cluster", size = "haut_tot")
fig_1 = px.box(data_selection, x = "cluster", y = "haut_tot")
fig_1.update_layout(title_text="Hauteur des arbres dans chaque cluster")
fig.show()
fig_1.show()

# Fonctionnalité supplémentaire : Détection des anomalies
# Recherche du meilleur eps
data_anomalies = data[["longitude", "latitude", "fk_prec_estim", "tronc_diam"]].copy()
data_anomalies_scaled = scaler.fit_transform(data_anomalies)
neighbors = NearestNeighbors(n_neighbors=10)
neighbors_fit = neighbors.fit(data_anomalies_scaled)
distances, indices = neighbors_fit.kneighbors(data_anomalies_scaled)

# Trier les distances pour tracer la "coude"
distances = np.sort(distances[:, 4], axis=0)
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.ylabel('Distance')
plt.xlabel('Points de données ordonnés')
plt.title('Graphique des distances des K-Plus-Proches-Voisins')
plt.show()

# Hauteur du tronc, haut_tot, tronc_diam, age_estim, fk_prec_estim
data_anomalies = data[["longitude", "latitude", "fk_prec_estim", "tronc_diam"]].copy()
data_anomalies_scaled = scaler.fit_transform(data_anomalies)
dbscan = DBSCAN(eps=1.5, min_samples=8)  # 2 à 4 fois le nombre de colonnes choisi
clusters = dbscan.fit_predict(data_anomalies_scaled)
data_anomalies['cluster'] = clusters
outliers = data_anomalies[data_anomalies['cluster'] == -1]
print("Number of outliers :", len(outliers))
plt.figure(figsize=(10, 13))
plt.scatter(data_anomalies['fk_prec_estim'], data_anomalies['tronc_diam'], c=data_anomalies['cluster'], cmap='coolwarm',
            label='Clusters')
plt.scatter(outliers['fk_prec_estim'], outliers['tronc_diam'], c='black', label='Outliers', marker='x')
plt.xlabel("Précision de l'âge estimé")
plt.ylabel("Diamètre du tronc")
plt.title(
    "Détection des Anomalies des Arbres avec DBSCAN, en fonction de la précision de l'âge estimé et du diamètre du tronc")
plt.colorbar(label='Cluster')
plt.show()

