"""
MQTT Subscriber - Écoute les demandes et effectue des prédictions batch
TeleConnect Afrique
"""

import paho.mqtt.client as mqtt
import json
import requests
import logging
import sqlite3
from datetime import datetime
import time
from typing import List, Dict, Any

# Configuration SQLite
DB_NAME = "customers.db" # Le même fichier que nous avions créé
TABLE_LOG = "predictions_log"

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration MQTT
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_REQUEST = "teleconnect/churn/prediction/request"
MQTT_TOPIC_RESPONSE = "teleconnect/churn/prediction/response"

# === Vos identifiants MQTT ===
MQTT_USER = "user"  # <-- VOS IDENTIFIANTS
MQTT_PASSWORD = "morose20" # <-- VOS IDENTIFIANTS
# ===============================

# Configuration API
API_URL = "http://localhost:8000/predict_batch"

print("="*80)
print("MQTT SUBSCRIBER - TeleConnect Churn Prediction")
print("="*80)

def initialize_db():
    """Crée la table de log des prédictions si elle n'existe pas."""
    CREATE_LOG_TABLE_QUERY = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_LOG} (
        customerID TEXT,
        churn_prediction TEXT,
        churn_probability REAL,
        risk_level TEXT,
        prediction_date TEXT, -- Sauvegarde au format ISO
        PRIMARY KEY (customerID, prediction_date) -- Pour suivre les réévaluations
    );
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(CREATE_LOG_TABLE_QUERY)
        conn.commit()
        logger.info(f"✅ Table de log '{TABLE_LOG}' initialisée dans '{DB_NAME}'")
    except sqlite3.Error as e:
        logger.error(f"❌ Erreur lors de l'initialisation de la BDD: {e}")
    finally:
        if conn:
            conn.close()

def log_predictions(predictions: List[Dict[str, Any]]):
    """Sauvegarde les résultats de la prédiction dans la base de données."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        log_data = []
        
        for p in predictions:
            log_data.append((
                p.get('customerID'),
                p.get('churn_prediction'),
                p.get('churn_probability'),
                p.get('risk_level'),
                timestamp
            ))

        INSERT_LOG_QUERY = f"""
        INSERT INTO {TABLE_LOG} 
        (customerID, churn_prediction, churn_probability, risk_level, prediction_date) 
        VALUES (?, ?, ?, ?, ?)
        """
        
        cursor.executemany(INSERT_LOG_QUERY, log_data)
        conn.commit()
        logger.info(f"💾 {len(log_data)} prédictions enregistrées.")
        
    except sqlite3.Error as e:
        logger.error(f"❌ Erreur lors de l'enregistrement des logs: {e}")
    finally:
        if conn:
            conn.close()


# Callback lors de la connexion
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info(f"✅ Connecté au broker MQTT ({MQTT_BROKER}:{MQTT_PORT})")
        client.subscribe(MQTT_TOPIC_REQUEST)
        logger.info(f"✅ Abonné au topic: {MQTT_TOPIC_REQUEST}")
        logger.info(f"⏳ En attente de requêtes de prédiction...")
    else:
        logger.error(f"❌ Échec de connexion, code: {rc}")


# Callback lors de la réception d'un message
def on_message(client, userdata, msg):
    logger.info(f"\n📨 Message reçu sur le topic: {msg.topic}")
    
    try:
        # Charger le payload JSON
        payload = json.loads(msg.payload.decode('utf-8'))
        customers_to_predict = payload.get("customers", [])
        
        if not customers_to_predict:
            logger.warning("⚠️ Payload vide. Ignoré.")
            return

        logger.info(f"🔍 Requête de prédiction pour {len(customers_to_predict)} clients...")

        # 1. Appeler l'API FastAPI
        start_api_call = time.time()
        api_response = requests.post(API_URL, json={"customers": customers_to_predict})
        end_api_call = time.time()
        
        api_response.raise_for_status() # Lève une exception si le statut n'est pas 200
        
        result_batch = api_response.json()
        predictions = result_batch.get('predictions', [])
        
        logger.info(f"✅ Prédiction réussie en {(end_api_call - start_api_call)*1000:.2f} ms.")
        
        # 2. Sauvegarder les résultats dans la BDD pour le Dashboard
        log_predictions(predictions)
        
        # 3. Publier la réponse sur le topic de réponse MQTT
        final_response = {
            "status": "success",
            "summary": result_batch.get('summary'),
            "predictions_count": len(predictions)
        }
        
        client.publish(MQTT_TOPIC_RESPONSE, json.dumps(final_response))
        logger.info(f"📤 Réponse envoyée sur {MQTT_TOPIC_RESPONSE}.")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erreur lors de l'appel API: {e}")
        error_response = {
            "error": "Erreur lors de l'appel à l'API de prédiction. Est-ce qu'elle est démarrée ?",
            "detail": str(e),
            "status": "failed"
        }
        client.publish(MQTT_TOPIC_RESPONSE, json.dumps(error_response))
        
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}", exc_info=True)
        error_response = {
            "error": str(e),
            "status": "failed"
        }
        client.publish(MQTT_TOPIC_RESPONSE, json.dumps(error_response))

# Créer le client MQTT
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "TeleConnectSubscriber")
# Définir les identifiants
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

client.on_connect = on_connect
client.on_message = on_message

# ============================================================================
# DÉMARRAGE DU SUBSCRIBER
# ============================================================================
# Initialiser la table de log avant de se connecter
initialize_db()

# Connexion au broker
logger.info(f"🔄 Connexion au broker MQTT...")
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    logger.info(f"✅ Connexion établie")
    logger.info(f"\n📡 Configuration:")
    logger.info(f"   Broker: {MQTT_BROKER}:{MQTT_PORT}")
    logger.info(f"   Topic écoute: {MQTT_TOPIC_REQUEST}")
    logger.info(f"   Topic réponse: {MQTT_TOPIC_RESPONSE}")
    logger.info(f"   API: {API_URL}")
    logger.info(f"\n🚀 Subscriber démarré ! (Ctrl+C pour arrêter)")
    logger.info(f"{'='*80}")
    
    # Boucle infinie pour écouter les messages
    client.loop_forever()
    
except KeyboardInterrupt:
    logger.info(f"\n\n⚠️  Arrêt...")
except Exception as e:
    logger.error(f"❌ Échec de connexion au broker: {e}")