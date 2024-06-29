# Projet : Modèle de Scoring de Crédit

## Contexte

Vous êtes Data Scientist chez "Prêt à dépenser", une société financière. L'entreprise souhaite développer un outil de scoring crédit pour évaluer la probabilité qu'un client rembourse son crédit.

## Objectifs

1. Déployer un modèle d'inférence sur une plateforme Cloud.
2. Créer un pipeline d’entraînement des modèles.
3. Évaluer et optimiser les modèles.
4. Mettre en œuvre le versioning du code.
5. Suivre et maintenir la performance du modèle en production.

## Structure du Repository

```plaintext
projet7/
├── csv_files/
│   ├── app_datas.csv
│   ├── app_datas_light.csv
│   ├── app_datas_light_very_light.csv
│   ├── application_test.csv
│   ├── application_train.csv
│   ├── bureau_balance.csv
│   ├── bureau.csv
│   ├── credit_card_balance.csv
│   ├── HomeCredit_columns_description.csv
│   ├── installments_payments.csv
│   ├── POS_CASH_balance.csv
│   ├── previous_application.csv
│   └── sample_submission.csv
├── misc/
│   ├── kaggle_nbs/
│   │   ├── aguiar/
│   │   │   └── lightgbm_with_simple_features.py
│   │   └── khoersen/
│   │       ├── 01_start-here-a-gentle-introduction.ipynb
│   │       ├── 02_introduction-to-manual-feature-engineering.ipynb
│   │       ├── 03_introduction-to-manual-feature-engineering-p2.ipynb
│   │       ├── 04_automated-feature-engineering-basics.ipynb
│   │       ├── 05_tuning-automated-feature-engineering-exploratory.ipynb
│   │       ├── 06_introduction-to-feature-selection.ipynb
│   │       └── 07_intro-to-model-tuning-grid-and-random-search.ipynb
│   └── recap.odt
├── models/
│   └── best_xgboost_model.pkl
├── notebooks/
│   ├── annexeP7.ipynb
│   ├── import_searcher.ipynb
│   ├── mlartifacts/
│       └── [expérimentations MLFlow et artifacts de modèles]
├── scripts/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_inference.py
│   └── requirements_searcher.py
└── README.md
```

## Outils Utilisés

- **MLFlow** : Tracking des expérimentations et stockage des modèles.
- **Git** : Versionning du code.
- **Github** : Stockage du code et intégration continue.
- **Github Actions** : Déploiement continu et automatisé.
- **FastAPI/Flask** : Création de l'API de prédiction.
- **Streamlit** : Interface de test de l'API.
- **Evidently** : Détection du data drift.

## Installation et Configuration

### Prérequis

- Python 3.8 ou plus
- Git
- MLFlow

### Installation

Cloner le repository et installer les dépendances :

```bash
git clone https://github.com/Stabadev/Projet7.git
cd Projet7
pip install -r requirements.txt
```

### MLFlow

Initialisez MLFlow pour le tracking des expérimentations :

```bash
mlflow ui
```

## Utilisation

### Exécution de l'API

A voir


### Interface Utilisateur

A voir

### Exécution des Tests

A voir

# Installation :

On suppose que vous possédez un VPS sur lequel vous avez installé la distribution Yunohost, avec un sous-domaine, pour lequel vous avez installé le certificat SSL. 
Ici, le sous-domaine est `https://projet7.rogues.fr`.

Sur ce sous-domaine, on installera l'application `my_webapp`.
Ici, il s'agit de la seconde installation de l'application `my_webapp`, donc les fichiers se trouvent dans le dossier `/var/www/my_webapp__2/www/`.
De plus, la documentation nous invite à gérer les paramètres de Nginx en créant un fichier `projet7.conf` dans le dossier `/etc/nginx/conf.d/projet7.rogues.fr/my_webapp__2.d`.

## Cloner le dépôt GitHub :

```sh
cd /var/www/my_webapp__2/www
git clone -b master https://github.com/Stabadev/Projet7.git
```

## Mettre à jour le contenu du fichier `requirements.txt` :

```plaintext
joblib
lightgbm
matplotlib
mlflow
numpy
pandas
seaborn
shap
DateTime
xgboost
```

## Installer les dépendances :

```sh
pip install -r requirements.txt
```

## Adapter des fichiers pour fonctionnement sur VPS

- Modifier le contenu du fichier `~/Projet7/src/app.py` (voir contenu ci-dessous joint).
- Modifier le contenu du fichier `~/Projet7/templates/index.html` (voir contenu ci-dessous).

## Lancer l'application Flask :

```sh
cd src
export FLASK_APP=app.py
python -m flask run --host=0.0.0.0 --port=5001
```

## Tester l'application Flask

### Tester la route principale 

```sh
curl http://localhost:5001/
```

### Tester la requête POST

```sh
curl -X POST -d "client_id=12345" http://localhost:5001/predict
```

## Lancer l'application avec Gunicorn : 

```sh
cd /var/www/my_webapp__2/www/Projet7/src
gunicorn --bind 0.0.0.0:8000 app:app
```

## Tester Gunicorn sur le VPS :

```sh
curl -X POST -d "client_id=12345" http://localhost:8000/site/projet7/predict
```

## Configurer Nginx

### Créer le fichier de configuration Nginx `projet7.conf`

```sh
cd /etc/nginx/conf.d/projet7.rogues.fr.d/my_webapp__2.d
sudo touch projet7.conf
```

### Éditer le fichier de configuration Nginx `projet7.conf`

```sh
sudo nano projet7.conf
```

### Code du fichier de configuration Nginx `projet7.conf`

```nginx
location /site/projet7/ {
    proxy_pass http://127.0.0.1:8000;  # Port où Gunicorn écoute
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # WebSocket support
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

## Tester la configuration Nginx : 

```sh
sudo nginx -t
```

## Redémarrer le service Nginx :

```sh
sudo systemctl reload nginx
```

L'application est alors disponible sur `https://projet7.rogues.fr/site/projet7/`.

-----

## Nouveau code pour le fichier `app.py` :

```python
# code_app.py ~/Projet7/src/app.py

from flask import Flask, Blueprint, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import shap
import os

# Chemin de base pour les fichiers
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(BASE_DIR, '../templates')
app = Flask(__name__, template_folder=template_dir)

bp = Blueprint('projet7', __name__, url_prefix='/site/projet7')

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

@bp.route('/')
def index():
    return render_template('index.html', prediction=None, local_importances=None, global_importances=global_importances_df)

@bp.route('/predict', methods=['POST'])
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

app.register_blueprint(bp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
```

-----

## Nouveau code pour le fichier `index.html` :

```html
# code_index.html dans le dossier ~/Projet7/templates

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Client Prediction</title>
    <style>
        .importance-table {
            border-collapse: collapse;
            width: 50%;
            margin-top: 20px;
        }
        .importance-table th, .importance-table td {
            border: 1px solid #dddddd;
            text-align: left;
            padding: 8px;
        }
        .importance-table th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <h1>Client Prediction</h1>
    <form action="/site/projet7/predict" method="post">
        <label for="client_id">Client ID:</label>
        <input type="text" id="client_id" name="client_id" required>
        <button type="submit">Predict</button>
    </form>

    {% if prediction is not none %}
        <h2>Prediction: {{ prediction }}</h2>
    {% endif %}

    {% if global_importances is not none %}
        <h2>Global Feature Importances</h2>
        <table class="importance

-table">
            <tr>
                <th>Feature</th>
                <th>Description</th>
                <th>Importance</th>
            </tr>
            {% for index, row in global_importances.iterrows() %}
            <tr>
                <td>{{ row['Feature'] }}</td>
                <td>{{ row['Description'] }}</td>
                <td>{{ row['Importance'] }}</td>
            </tr>
            {% endfor %}
        </table>
    {% endif %}

    {% if local_importances is not none %}
        <h2>Local Feature Importances</h2>
        <table class="importance-table">
            <tr>
                <th>Feature</th>
                <th>Description</th>
                <th>Importance</th>
            </tr>
            {% for index, row in local_importances.iterrows() %}
            <tr>
                <td>{{ row['Feature'] }}</td>
                <td>{{ row['Description'] }}</td>
                <td>{{ row['Importance'] }}</td>
            </tr>
            {% endfor %}
        </table>
    {% endif %}
</body>
</html>
```

## Contribuer

Les contributions sont les bienvenues ! Veuillez suivre les étapes suivantes pour contribuer :

1. Forker le projet.
2. Créer une branche feature (`git checkout -b feature/FeatureName`).
3. Committer vos changements (`git commit -m 'Add some FeatureName'`).
4. Pusher vers la branche (`git push origin feature/FeatureName`).
5. Ouvrir une Pull Request.

## Auteurs et Contributeurs

- **Alexandre ROGUES** - Créateur du projet
- **ChatGPT** - Co-auteur et support

## Contact

Pour toute question, vous pouvez me contacter à [alexandre.rogues@gmail.com](mailto:alexandre.rogues@gmail.com).
