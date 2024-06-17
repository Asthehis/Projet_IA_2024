import sklearn
import csv
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy

arbre = pd.read_csv("Data_Arbre.csv")
#print(arbre)

# Méthode regression Random Forest, CART...Gini index

x = arbre[['haut_tronc', 'fk_prec_estim', 'tronc_diam', 'haut_tot', 'fk_stadedev']].copy()
y = arbre[['age_estim']].copy
print(x)
print(y)

encoder = OrdinalEncoder()
x['fk_stadedev'] = encoder.fit_transform(x[['fk_stadedev']])
print(x['fk_stadedev'].head())

x_norm = (x - x.min()) / (x.max() - x.min())
print(x_norm)

