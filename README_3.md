BESOIN CLIENT 3

Fonctionnement du programme final :

– Le programme importe les données du fichier Notebook et lit un fichier JSON.

– Il vérifie que toutes les données étudiées soient présentes et bien encodées.

– Il filtre les données de fk_arb_état pour ne prendre que Non essouché et Essouché.

– Il normalise les données longitude, latitude, haut-tot, tronc-diam, haut-tronc et age_estim.

– Il effectue les prédictions des modèles RandomForest et GradientBoostingClassifier.
Le programme affiche ensuite la répartition des arbres sur une carte en fonction des prédictions du RandomForest.

Pour utiliser le script final du besoin client 3 :

– Ouvrir le fichier Script3.

– Lancer le code Python de ce script.

– Le programme s'exécutera.

En sortie, dans la console, vous obtiendrez les résultats des prédictions.

Vous obtiendrez également une carte qui représente les arbres susceptibles d'être arrachés en cas de tempête en fonction des prédictions.


