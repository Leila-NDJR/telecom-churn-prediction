# 📊 Système de Prédiction du Churn Client - TeleConnect Afrique

**Auteur : K. Jessy**
Projet End-to-End Data Science | M1 Data Science | Date : Novembre 2025

📝 Description du Projet
Ce projet est une solution complète (de la donnée brute à la visualisation) visant à prédire le désabonnement (churn) des clients de l'opérateur TeleConnect Afrique.

Face à un taux de churn de 27%, ce système permet de :

1- Analyser les données historiques pour comprendre les facteurs de risque.

2- Prédire en temps réel (via Batch) la probabilité de départ de nouveaux clients.

3- Visualiser les résultats sur un tableau de bord pour la prise de décision.

L'architecture repose sur un modèle XGBoost exposé via une API FastAPI, communiquant via un protocole de messagerie asynchrone (MQTT) et monitoré via Streamlit.

📂 Structure du Projet
Voici l'organisation des fichiers du projet :

Bash

telecom-churn-prediction/
│
├── api/                       # COEUR DU SYSTÈME (BACKEND)
│   ├── assets/                # Ressources statiques (optionnel)
│   ├── main.py                # Application FastAPI (Logique de prédiction & Endpoints)
│   └── test_api.py            # Script de test de performance et de latence
│
├── data/                      # GESTION DES DONNÉES
│   ├── processed/             # Artefacts du modèle (Ne pas supprimer/modifier)
│   │   ├── best_model.pkl     # Modèle XGBoost entraîné
│   │   ├── scaler.pkl         # Préprocesseur (StandardScaler + OHE)
│   │   ├── model_metadata.json
│   │   └── ...
│   └── raw/                   # Données sources (CSV Kaggle)
│
├── monitoring/                # SURVEILLANCE ET MESSAGING
│   ├── dashboard.py           # Interface Utilisateur (Streamlit)
│   ├── mqtt_publisher.py      # Simulateur d'envoi de clients (Source)
│   └── mqtt_suscriber.py      # Service d'écoute et d'enregistrement en BDD
│
├── notebooks/                 # LABORATOIRE DATA SCIENCE (Jupyter)
│   ├── 01_eda.ipynb                # Analyse exploratoire des données
│   ├── 02_preprocessing.ipynb      # Nettoyage et Feature Engineering
│   ├── 03_modeling.ipynb           # Entraînement, tuning et évaluation
│   └── 04_business_analysis.ipynb  # Calcul du ROI et analyse d'impact
│
├── customers.db               # Base de données SQLite active (Clients + Logs)
├── customers_db.py            # Script 1 : Initialisation de la BDD (10 clients)
├── add_new_customers_db.py    # Script 2 : Ajout de données de test (30 clients)
├── requirements.txt           # Liste des dépendances Python
└── README.md                  # Documentation officielle


⚙️ Prérequis Techniques
Avant de commencer, assurez-vous que votre machine dispose de :

Python 3.10+ installé.

Eclipse Mosquitto (Broker MQTT) installé et démarré.

Windows : Télécharger sur mosquitto.org. Une fois installé, lancez le service (Services Windows -> Mosquitto Broker) ou exécutez mosquitto -v dans un terminal.


🚀 Guide d'Installation (Pas à Pas)

1. Cloner et configurer l'environnement (git clone https://github.com/Leila-NDJR/telecom-churn-prediction.git )

Ouvrez votre terminal (PowerShell ou Bash) à la racine du projet :

# 1. Créer un environnement virtuel pour isoler le projet
python -m venv venv

# 2. Activer l'environnement
# Sur Windows :
venv\Scripts\activate
# Sur Mac/Linux :
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

2. Initialiser la Base de Données
Nous devons créer la base de données locale SQLite et y injecter des clients pour simuler un environnement réel.

Note : Si un fichier customers.db existe déjà et que vous voulez repartir de zéro, supprimez-le avant de lancer ces commandes.

# Étape A : Créer la structure et les 10 premiers clients
python customers_db.py

# Étape B : Ajouter 30 clients supplémentaires pour le test batch
python add_new_customers_db.py
▶️ Guide d'Exécution (Démarrage du Système)
Pour voir le projet fonctionner, vous devez ouvrir 4 terminaux différents (avec l'environnement virtuel activé dans chacun) et lancer les services dans cet ordre précis.

Terminal 1 : L'API de Prédiction (Le Cerveau)
Ce service charge le modèle XGBoost et attend les requêtes.

Bash 

python api/main.py
Attendre le message : Application startup complete. L'API est accessible sur : http://localhost:8000/docs

Terminal 2 : Le Subscriber MQTT (Le Pont)
Ce service écoute les demandes, interroge l'API et sauvegarde les résultats.

Bash

python monitoring/mqtt_suscriber.py
Attendre le message : ✅ Connecté au broker MQTT

Terminal 3 : Le Dashboard (La Vue)
L'interface pour visualiser les KPIs et les alertes churn.

Bash

streamlit run monitoring/dashboard.py
Votre navigateur s'ouvrira automatiquement sur : http://localhost:8501

Terminal 4 : Le Publisher MQTT (Le Déclencheur)
Ce script simule l'envoi d'un lot de clients (batch) à analyser.

Bash

python monitoring/mqtt_publisher.py
✅ Vérification du Bon Fonctionnement
Si tout s'est bien passé :

Dans le Terminal 4 (Publisher), vous voyez : 📤 Envoi de 40 clients...

Dans le Terminal 2 (Subscriber), vous voyez défiler les logs : 📦 Batch reçu, ✅ Prédiction réussie, 💾 40 prédictions enregistrées.

Sur le Dashboard (Navigateur), rafraîchissez la page (ou attendez 60s). Vous devriez voir :

Statistiques du Dernier Batch : 40 clients.

Tableau détaillé : La liste des clients avec leur ID, leur probabilité de churn et le niveau de risque.

# 📊 Performances et Métriques

Le modèle a été évalué sur un jeu de test indépendant.

Modèle utilisé : XGBoost Classifier.

Seuil de décision : 0.30 (Optimisé business — voir `data/processed/business_analysis.json`).
Ce seuil maximise le revenu net attendu (analyse coûts/bénéfices Notebook 04) : ~1 113 400 FCFA à 0.30 contre ~1 065 000 FCFA au seuil générique 0.50, soit un gain de ~48 400 FCFA sur le jeu de test.

AUC-ROC : 0.84

Recall (Détection des partants) : 80.7% au seuil générique 0.50, **92.5% au seuil business 0.30**.

Latence API (Batch 30 clients) : ~2600ms.

## ✅ Résultats clés

Le seuil de décision de l'API était figé à une valeur générique (0.50) sans jamais lire l'analyse ROI déjà calculée dans le Notebook 04. Après correction :

- **Seuil business 0.30 réellement branché en production** (`api/main.py` lit désormais `business_analysis.json`, avec repli sur `model_metadata.json` puis 0.35).
- **+4,5 % de revenu net attendu** sur le jeu de test (1 407 clients) : ~1 113 400 FCFA au seuil 0.30 contre ~1 065 000 FCFA au seuil générique 0.50 (soit +48 400 FCFA).
- **Recall porté à 92,5 %** (contre 80,7 % au seuil 0.50) : beaucoup plus de clients à risque réellement détectés, au prix d'un peu plus de faux positifs — arbitrage assumé par l'analyse coûts/bénéfices.
- **Score de priorisation `priority_score` = risque × valeur client** (`churn_probability × clv_proxy`), exposé dans `/predict` et `/predict_batch`, pour cibler en premier les clients à la fois à risque ET à forte valeur avec un budget de rétention limité.
- **Recommandations enfin actionnables** : une seule fonction `get_recommendations()` (au lieu de deux copies dupliquées), avec canal de contact adapté au profil et offre chiffrée réelle.

## 🎯 Recommandations

À l'attention d'une direction rétention / CRM souhaitant s'appuyer sur ce système :

1. **Valider la transférabilité du modèle sur de vraies données TeleConnect Afrique avant tout déploiement réel** — le dataset d'entraînement (Kaggle, IBM Telco) est d'origine occidentale (marché nord-américain) ; les structures tarifaires et comportements de churn africains peuvent différer sensiblement.
2. **Mesurer l'efficacité réelle des actions de rétention par un test A/B** avant toute généralisation — comparer un groupe de clients contactés selon `priority_score` (canal + offre) à un groupe témoin, sur 1 à 2 cycles de facturation.
3. **Réviser le seuil business (0.30) trimestriellement**, en le recalculant sur `business_analysis.json` à mesure que de nouvelles données de churn et de coûts de rétention arrivent, plutôt que de le figer durablement dans le code.
4. **Doter les équipes terrain d'un tableau de bord trié par `priority_score`** pour allouer en priorité le budget de rétention limité aux clients à la fois les plus à risque et les plus rentables.