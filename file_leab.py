import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data=pd.read_csv(r"C:\Users\leabo\Desktop\ISEN 3 ème année\projet\Data_Arbre.csv")
#print(data.head())

data_selected = data[['longitude', 'latitude', 'haut_tot', 'tronc_diam']].copy()

#print(data_selected.head())


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected[['haut_tot', 'tronc_diam']])

silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, labels)
    silhouette_scores.append(score)

plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de silhouette')
plt.title('Score de silhouette pour différents nombres de clusters')
plt.show()
