📊 Système de Prédiction du Churn Client - TeleConnect Afrique
M2 Data Science - Projet End-to-End Auteur : K. Jessy | Date : Novembre 2025

📝 Contexte du Projet
TeleConnect Afrique, opérateur majeur en Afrique de l'Ouest, fait face à un taux de désabonnement (churn) critique de 27%. Ce projet vise à déployer une solution Data Science complète pour prédire quels clients risquent de quitter l'opérateur afin de mener des actions de rétention ciblées.

L'objectif technique est de fournir un système capable de traiter des lots de clients (batch processing) via une architecture découplée et performante, avec un monitoring en temps réel.

🏗️ Architecture Technique
Le système repose sur une architecture événementielle asynchrone :

Source de Données (SQLite) : Stockage des informations clients.

Publisher (MQTT) : Simule l'envoi de lots de clients à analyser.

Broker (Mosquitto) : Gère la file d'attente des messages.

Subscriber (MQTT) : Écoute les demandes, interroge l'API et stocke les résultats.

API de Prédiction (FastAPI) : Expose le modèle XGBoost optimisé (latence < 200ms).

Dashboard (Streamlit) : Visualisation des KPIs et des clients à risque.

⚙️ Stack Technique
Langage : Python 

API & Web : FastAPI, Uvicorn

Machine Learning : XGBoost, Scikit-learn, Pandas, NumPy

Messaging : Eclipse Mosquitto, Paho-MQTT

Visualisation : Streamlit, Plotly

Base de Données : SQLite3

📂 Structure du Projet
## 📂 Structure du Projet


telecom-churn-prediction/
│
├── api/                       # API FastAPI
│   ├── main.py                # Application principale (Endpoints & Logique)
│   └── test_api.py            # Script de test de performance
│
├── data/                      # Gestion des données
│   ├── processed/             # Artefacts générés (Modèles, Scalers, Métadonnées)
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   ├── model_metadata.json
│   │   └── ...
│   └── raw/                   # Données brutes
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── monitoring/                # Surveillance temps réel
│   ├── dashboard.py           # Interface Streamlit
│   ├── mqtt_publisher.py      # Simulateur d'envoi de clients
│   └── mqtt_suscriber.py      # Service d'écoute et d'enregistrement
│
├── notebooks/                 # Étapes de Data Science (Jupyter)
│   ├── 01_eda.ipynb                # Analyse exploratoire
│   ├── 02_preprocessing.ipynb      # Nettoyage et Feature Engineering
│   ├── 03_modeling.ipynb           # Entraînement et évaluation
│   └── 04_business_analysis.ipynb  # Analyse d'impact business
│
├── src/                       # Code source modulaire
│   ├── data_processing/       # Scripts de traitement de données
│   ├── models/                # Classes de modèles
│   └── utils/                 # Fonctions utilitaires
│
├── customers.db               # Base de données SQLite active
├── customers_db.py            # Script d'initialisation de la BDD
├── add_new_customers_db.py    # Script d'ajout de données de test
├── requirements.txt           # Liste des dépendances Python
└── README.md                  # Documentation du projet
🚀 Guide d'Installation et de Démarrage
1. Prérequis
Assurez-vous d'avoir Python installé ainsi que le broker Mosquitto en cours d'exécution sur votre machine.

2. Installation des dépendances
Bash

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows :
venv\Scripts\activate

# Installer les paquets requis
pip install -r config/requirements.txt

3. Initialisation de la Base de Données
Avant de lancer le système, nous devons créer la base de données et y injecter des clients fictifs.

Bash

# Supprimer l'ancienne base si elle existe pour partir sur du propre
# del customers.db (Windows) ou rm customers.db 

# Créer la base et les 10 premiers clients
python customers_db.py

# Ajouter 30 clients supplémentaires pour le test
python add_new_customers_db.py

4. Lancement des Services
Ouvrez 4 terminaux différents pour lancer les composants du système :

Terminal 1 : L'API de Prédiction

Bash

python api/main.py
# L'API sera accessible sur http://localhost:8000
Terminal 2 : Le Subscriber (Écouteur)

Bash

python monitoring/mqtt_suscriber.py
# Attend les messages MQTT...
Terminal 3 : Le Dashboard de Monitoring

Bash

streamlit run monitoring/dashboard.py
# Ouvre le navigateur sur http://localhost:8501
Terminal 4 : Le Publisher (Déclencheur)

Bash

python monitoring/mqtt_publisher.py
# Envoie le lot de 40 clients pour analyse
📊 Performances du Modèle
Le modèle utilisé est un XGBoost Classifier optimisé.

Métriques Techniques (Test Set)
AUC-ROC : 0.84

Recall (Taux de détection) : 80.7% (Priorité projet : minimiser les Faux Négatifs)

Accuracy : 74.1%

API : > 200 ms 

Métriques Business
Seuil de décision : 0.50 (Optimisé pour l'équilibre Précision/Rappel)

Niveaux de Risque :

🔴 High (> 75%) : Action immédiate requise.

🟠 Medium (50-75%) : Campagne de rétention standard.

🟢 Low (< 50%) : Client stable.

🛠️ Fonctionnalités Clés Implémentées
Feature Engineering Automatisé :

Création dynamique de variables (TotalServices, AvgMonthlyCharges, SeniorWithFamily) directement dans l'API.

Pipeline de Prétraitement Robuste :

Gestion des valeurs manquantes.

One-Hot Encoding aligné avec le modèle d'entraînement.

Scaling des données.

Historisation et Dédoublonnage :

Le système garde une trace unique de la dernière prédiction pour chaque client (INSERT OR REPLACE dans SQLite).

Monitoring Temps Réel :

Le dashboard se met à jour automatiquement à chaque nouveau batch traité.