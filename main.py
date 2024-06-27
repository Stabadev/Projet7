import os
import joblib
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

template_dir = os.path.abspath('FastAPIAppFolder/templates')
app = FastAPI()
templates = Jinja2Templates(directory=template_dir)

# Vérifiez que les chemins sont corrects
model_path = "FastAPIAppFolder/models/best_xgboost_model.pkl"
client_data_path = 'FastAPIAppFolder/csv_files/app_datas_light_very_light.csv'
descriptions_path = 'FastAPIAppFolder/csv_files/HomeCredit_columns_description.csv'

# Charger le modèle
try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

# Charger les données clients une fois au démarrage de l'application
try:
    client_data_df = pd.read_csv(client_data_path)
except Exception as e:
    print(f"Failed to load client data: {e}")
    raise

# Essayer de charger les descriptions des caractéristiques avec différents encodages
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
descriptions_df = None
for encoding in encodings:
    try:
        descriptions_df = pd.read_csv(descriptions_path, usecols=['Row', 'Description'], encoding=encoding)
        break
    except UnicodeDecodeError as e:
        print(f"Failed to read with encoding {encoding}: {e}")

if descriptions_df is None:
    raise Exception("Failed to load descriptions with all attempted encodings.")

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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("fastapi_index.html", {"request": request, "prediction": None, "local_importances": None, "global_importances": global_importances_df.to_dict(orient='records')})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, client_id: int = Form(...)):
    client_data = get_client_data(client_id)
    if client_data is None:
        return templates.TemplateResponse("fastapi_index.html", {"request": request, "prediction": "Client ID not found", "local_importances": None, "global_importances": global_importances_df.to_dict(orient='records')})

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

    return templates.TemplateResponse("fastapi_index.html", {
        "request": request,
        "prediction": int(prediction[0]),
        "local_importances": local_importances_df.to_dict(orient='records'),
        "global_importances": global_importances_df.to_dict(orient='records')
    })
