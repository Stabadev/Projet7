from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import shap
import os

# Chemin de base pour les fichiers
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(BASE_DIR, '../templates')
app = Flask(__name__, template_folder=template_dir)

# Charger le modèle
model_path = os.path.join(BASE_DIR, "../models/best_xgboost_model.pkl")
model = joblib.load(model_path)

# Charger les données clients une fois au démarrage de l'application
client_data_path = os.path.join(BASE_DIR, '../csv_files/app_datas_light_imputed_scaled.csv')
client_data_df = pd.read_csv(client_data_path)

# Supprimer les colonnes 'TARGET' et 'SK_ID_CURR' pour obtenir les mêmes features que celles utilisées pour entraîner le modèle
features_df = client_data_df.drop(columns=['SK_ID_CURR', 'TARGET'])

# Essayer de charger les descriptions des caractéristiques avec différents encodages
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
descriptions_df = None
for encoding in encodings:
    try:
        descriptions_path = os.path.join(BASE_DIR, '../csv_files/HomeCredit_columns_description.csv')
        descriptions_df = pd.read_csv(descriptions_path, usecols=['Row', 'Description'], encoding=encoding)
        break
    except UnicodeDecodeError as e:
        print(f"Failed to read with encoding {encoding}: {e}")

if descriptions_df is None:
    raise ValueError("Failed to read the descriptions file with all attempted encodings.")

descriptions_dict = descriptions_df.set_index('Row')['Description'].to_dict()

# Calculer les importances globales des caractéristiques
global_feature_importances = model.feature_importances_
features = features_df.columns

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
    # Supprimer la colonne 'SK_ID_CURR' et 'TARGET' pour obtenir uniquement les caractéristiques du modèle
    client_data = client_data.drop(columns=['SK_ID_CURR', 'TARGET'])
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
    app.run(debug=True, host='0.0.0.0', port=5001)
