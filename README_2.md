BESOIN CLIENT 2

Fonctionnement du programme final :

- Le programme importe les modèles du fichier Notebook 
et lit un fichier JSON.

- Il vérifie que toutes les données étudiées soient 
présentes et bien encodées.

- Si les données ne sont pas bien encoder, il utilise l'Ordinal Encoder
et le Label Encoder pour les colonnes "fk_stadedev" et "fk_nomtech"

- Le proggramme sélectionne ensuite les données "haut_tronc", 
"fk_stadedev", "tronc_diam", "haut_tot", "fk_nomtech" et "age_estim", 
puis les normalises

- Il effectue ensuite les prédictions des données normalisées à l'aide des modèles RandomForest,
DecisionTreeRegressor, GradientBoostingRegression, KNeighborsRegressor
et ExtraTreesRegressor

- Il effectue ensuite les calcul métriques à l'aide de "mean_squared_error"
et de "r2_score"

- Il "dénormalise" ensuite les données et créé un dictionnaire qui contiendra
les prédictions pour chaque modèle

- Enfin, il créé un dataframe de ces prédictions et les sauvegardes en format JSON



Pour utiliser le script final du besoin client 3 :

- Ouvrir le fichier Script2.py.

- Donner un fichier en format JSON, dans l'appel de la fonction, à la fin du fichier

- Lancer le code Python de ce script.

- Le programme s'exécutera.

- Ouvrir le fichier JSON créé, nommé "prediction2.json"


En sortie, dans la console, vous obtiendrez la moyenne des erreur carrées
(Mean Squared Error), ainsi que la valeur des R² pour chaque modèle.

Vous obtiendrez également un fichier JSON qui condiendra les prédictions
des âges estimés.