import numpy as np
import pandas as pd


data=pd.read_csv(r"C:\Users\leabo\Desktop\ISEN 3 ème année\projet\Data_Arbre.csv")

data_selected = data[['longitude', 'latitude', 'haut_tot', 'tronc_diam', 'haut_tronc', 'age_estim']].copy()

print(data_selected)


