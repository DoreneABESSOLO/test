import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

df = pd.read_csv('C:/Users/bamiy/OneDrive/Documents/chasseurs/houses_Madrid_cleaned.csv')

#caractéristiques pertinentes
carateristiques = [
    'sq_mt_built', 
    'n_rooms', 
    'n_bathrooms', 
    'floor', 
    'has_lift', 
    'is_exterior', 
    'has_parking'
]

# caracteristique et cible
X = df[carateristiques]
y = df['buy_price']  

#gestion des valeurs manquantes
X = X.fillna(0) 
y = y.fillna(y.mean()) 

#encodage des variables catégorielles
#X = pd.get_dummies(X, drop_first=True)

#ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#entraînement du modèle de régression linéaire multiple
model = LinearRegression()
model.fit(X_train, y_train)

#prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

#evaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# résultats d'évaluation
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

#exemple d'utilisation du modèle pour prédire le prix d'un nouveau bien
nouveau_bien = [[100, 3, 2, 2, 1, 1, 0]]  #exemple: 100m², 3 pièces, 2 salles de bain, étage 2, ascenseur, extérieur, pas de parking
prix_estime = model.predict(nouveau_bien)

print(f"Le prix estimé du nouveau bien est : {prix_estime[0]}")
