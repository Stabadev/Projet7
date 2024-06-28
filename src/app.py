from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import shap
import os

template_dir = os.path.abspath('../templates')
app = Flask(__name__, template_folder=template_dir)

# Charger le modèle
model = joblib.load("../models/best_xgboost_model.pkl")

# Charger les données clients une fois au démarrage de l'application
client_data_df = pd.read_csv('../csv_files/app_datas_light_imputed_scaled.csv')

# Essayer de charger les descriptions des caractéristiques avec différents encodages
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
for encoding in encodings:
    try:
        descriptions_df = pd.read_csv('../csv_files/HomeCredit_columns_description.csv', usecols=['Row', 'Description'], encoding=encoding)
        break
    except UnicodeDecodeError as e:
        print(f"Failed to read with encoding {encoding}: {e}")
descriptions_dict = descriptions_df.set_index('Row')['Description'].to_dict()


# Calculer les importances globales des caractéristiques
global_feature_importances = model.feature_importances_
features = client_data_df.drop(columns=['SK_ID_CURR']).columns

# Créer un DataFrame pour les importances globales avec les descriptions
global_importances_df = pd.DataFrame({
    'Feature': features,
    'Importance': global_feature_importances
})
global_importances_df['Description'] = global_importances_df['Feature'].map(descriptions_dict)
global_importances_df = global_importances_df.sort_values(by='Importance', ascending=False).head(5)

def get_client_data(client_id):
    # Rechercher les données du client spécifique
    client_data = client_data_df[client_data_df['SK_ID_CURR'] == client_id]
    if client_data.empty:
        return None
    # Supprimer la colonne 'SK_ID_CURR' pour obtenir uniquement les caractéristiques du modèle
    client_data = client_data.drop(columns=['SK_ID_CURR'])
    # Convertir les données en format nécessaire pour le modèle
    return client_data.values.flatten()

@app.route('/')
def index():
    return render_template('index.html', prediction=None, local_importances=None, global_importances=global_importances_df)

@app.route('/predict', methods=['POST'])
def predict():
    client_id = int(request.form['client_id'])
    client_data = get_client_data(client_id)
    if client_data is None:
        return render_template('index.html', prediction='Client ID not found', local_importances=None, global_importances=global_importances_df)
    
    prediction = model.predict([client_data])
    
    # Calculer les importances locales des caractéristiques
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values([client_data])
    
    local_importances_df = pd.DataFrame({
        'Feature': features,
        'Importance': shap_values[0]
    })
    local_importances_df['Description'] = local_importances_df['Feature'].map(descriptions_dict)
    local_importances_df = local_importances_df.sort_values(by='Importance', ascending=False).head(5)
    
    return render_template('index.html', prediction=int(prediction[0]), local_importances=local_importances_df, global_importances=global_importances_df)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
