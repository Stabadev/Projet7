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
