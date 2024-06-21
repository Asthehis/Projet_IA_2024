Besoin client 1 :
-


Fonctionnement du programme final : 
- Le programme commence par vous demandez le nombre de catégories souhaité
- Il importe ensuite un fichier csv puis sélectionne les colonnes de ce fichier 
qui sont utiles pour la réalisation de ce besoin (longitude, latitude et hauteur totale)
- Il normalise ensuite les données composant la colonne de la hauteur totale
- Grâce à l'algorithme K-Means, il prédit ensuite les valeurs de cette hauteur totale normalisée
et calcule son silhouette score
- Le programme affiche ensuite la répartition des arbres sur une carte, en fonction 
de la hauteur des arbres, ainsi qu'un boxplot détaillant la hauteur des arbres de chaque
boxplot
- Il détecte ensuite les anomalies et affiche trois graphiques matplotlib : 
précision de l'âge estimé / diamètre du tronc, hauteur totale / diamètre du tronc, et 
âge estimé / diamètre du tronc

Pour utiliser le script final du besoin
client 1 : 
- ouvrir le  fichier Besoin_client_1_script_final_py.
- Lancer ensuite le code python de ce script
- Dans la console, entrer le nombre de cluster
souhaité
- Le programme s'éxécutera 

En sortie, dans la console, vous obtiendrez 
l'en-tête du tableau, la prédiction à
l'aide de l'algorithme K-Means, le silhouette
score de cet algorithme et le nombre d'anomalies pour chaque graphique réalisé

Vous obtiendrez également une carte 
représentant la répartition des arbres
dans les clusters en fonction de la hauteur, un
boxplot indiquant la hauteur des arbres 
dans chaque cluster, et trois graphiques
représentant les anomalies précision de l'age/diamètre, 
hauteur totale/diamètre, et age/diamètre
