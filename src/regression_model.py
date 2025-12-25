import pandas as pd
from sklearn.linear_model import LinearRegression

def get_trained_model():
    """
    Crée et entraîne un modèle de régression linéaire sur des données d'exemple.
    """
    # 1. Créer des données d'exemple (surface en m² et prix en euros)
    data = {
        'surface': [60, 75, 80, 100, 120, 150],
        'prix': [200000, 250000, 260000, 320000, 380000, 450000]
    }
    df = pd.DataFrame(data)

    # 2. Définir les caractéristiques (X) et la cible (y)
    X = df[['surface']]
    y = df['prix']

    # 3. Créer et entraîner le modèle de régression linéaire
    model = LinearRegression()
    model.fit(X, y)
    
    return model

def predict_price(model, surface):
    """
    Prédit le prix pour une nouvelle maison étant donné sa surface.
    """
    # 4. Prédire le prix pour la surface de la nouvelle maison
    nouvelle_maison_surface = [[surface]]
    predicted_price = model.predict(nouvelle_maison_surface)

    return predicted_price[0]

# Cette partie est pour le test autonome du module
if __name__ == '__main__':
    model = get_trained_model()
    prediction = predict_price(model, 90)
    print(f"Le prix prédit pour une maison de 90m² est de : {prediction:.2f} euros")