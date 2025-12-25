import os

from flask import Flask, send_file, request, jsonify
from src.regression_model import get_trained_model, predict_price

app = Flask(__name__)

# Entraîner le modèle au démarrage de l'application
model = get_trained_model()

@app.route("/")
def index():
    return send_file('src/index.html')

@app.route("/predict")
def predict():
    # Obtenir la surface depuis les paramètres de la requête
    surface = request.args.get('surface', type=float)

    if surface is None:
        return jsonify({'error': 'Le paramètre surface est manquant'}), 400

    # Prédire le prix en utilisant le modèle chargé
    predicted_price = predict_price(model, surface)

    # Retourner la prédiction au format JSON
    return jsonify({'predicted_price': round(predicted_price, 2)})

def main():
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()
