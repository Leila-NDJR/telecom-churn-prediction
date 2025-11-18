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

# Configuration SQLite
DB_NAME = "customers.db" # Le même fichier que nous avions créé

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

# === AJOUTEZ CES DEUX LIGNES ===
MQTT_USER = "user"  # <-- REMPLACEZ PAR VOTRE VRAI USER 
MQTT_PASSWORD = "morose20" # <-- REMPLACEZ PAR VOTRE VRAI MOT DE PASSE
# ===============================

# Configuration API
API_URL = "http://localhost:8000/predict_batch"

print("="*80)
print("MQTT SUBSCRIBER - TeleConnect Churn Prediction")
print("="*80)

def initialize_db():
    """Crée la table de log des prédictions si elle n'existe pas,
    en s'assurant que customerID est UNIQUE."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        logger.info("🔧 Création/Vérification de la table 'predictions_log'...")
        # Ligne Modifiée pour inclure UNIQUE(customerID)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customerID TEXT UNIQUE,  -- <<< MODIFICATION CLÉ 1
            churn_prediction TEXT,
            churn_probability REAL,
            risk_level TEXT,
            batch_timestamp REAL,
            prediction_date TEXT
        );
        """)
        conn.commit()
        logger.info(f"✅ Table 'predictions_log' vérifiée/créée (customerID est UNIQUE) dans {DB_NAME}")
    except sqlite3.Error as e:
        logger.error(f"❌ Erreur SQLite lors de l'initialisation: {e}")
    finally:
        if conn:
            conn.close()

# Appelez cette fonction une fois pour être sûr que la table existe
initialize_db()

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
    logger.info(f"📩 MESSAGE REÇU sur {msg.topic}")
    logger.info("-"*80)

    conn = None
    
    try:
        # Décoder le message et extraire les données
        request_data = json.loads(msg.payload.decode())
        customers = request_data.get('customers', [])
        # 'timestamp' est le batch_timestamp
        batch_timestamp = request_data.get('timestamp', time.time())
        
        logger.info(f"📦 Batch reçu: {len(customers)} clients")
        logger.info(f"🔄 Envoi à l'API pour prédiction...")
        
        api_payload = {"customers": customers}
        # Appel API
        response = requests.post(API_URL, json=api_payload, timeout=30)
        
        if response.status_code == 200:
            predictions_result = response.json()
            predictions = predictions_result.get('predictions', [])
            
            logger.info(f"✅ Prédictions réussies ! ({len(predictions)} résultats)")
            
            # === DÉBUT DE LA SAUVEGARDE SQLITE (POINT B.) ===
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            data_to_insert = []
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for pred in predictions:
                # IMPORTANT : Extraction de l'ID client pour l'unicité
                customer_id = pred.get('customerID', 'UNKNOWN') 

                data_to_insert.append((
                    customer_id,
                    pred['churn_prediction'], 
                    pred['churn_probability'],
                    pred['risk_level'],
                    batch_timestamp,
                    current_datetime
                ))

            # Requête pour insérer ou remplacer (UPSERT)
            insert_query = """
            INSERT OR REPLACE INTO predictions_log 
            (customerID, churn_prediction, churn_probability, risk_level, batch_timestamp, prediction_date) 
            VALUES (?, ?, ?, ?, ?, ?);
            """
            
            cursor.executemany(insert_query, data_to_insert)
            conn.commit()
            
            logger.info(f"💾 {cursor.rowcount} prédictions mises à jour/ajoutées dans '{DB_NAME}'")
            # === FIN DE LA SAUVEGARDE SQLITE ===
            
        else:
            logger.error(f"❌ Erreur API: {response.status_code}")
            error_response = {
                "error": f"API error: {response.status_code}",
                "detail": response.text
            }
            client.publish(MQTT_TOPIC_RESPONSE, json.dumps(error_response))
        
        logger.info("-"*80)
        logger.info(f"⏳ En attente de nouvelles requêtes...\n")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement: {e}", exc_info=True)
        error_response = {
            "error": str(e),
            "status": "failed"
        }
        client.publish(MQTT_TOPIC_RESPONSE, json.dumps(error_response))

# Créer le client MQTT
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "TeleConnectSubscriber")

# === AJOUTEZ CETTE LIGNE ===
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
# ===========================

client.on_connect = on_connect
client.on_message = on_message

# Connexion au broker
logger.info(f"🔄 Connexion au broker MQTT...")
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    logger.info(f"✅ Connexion établie")
    logger.info(f"")
    logger.info(f"📡 Configuration:")
    logger.info(f"   Broker: {MQTT_BROKER}:{MQTT_PORT}")
    logger.info(f"   Topic écoute: {MQTT_TOPIC_REQUEST}")
    logger.info(f"   Topic réponse: {MQTT_TOPIC_RESPONSE}")
    logger.info(f"   API: {API_URL}")
    logger.info(f"")
    logger.info(f"🚀 Subscriber démarré ! (Ctrl+C pour arrêter)")
    logger.info(f"="*80)
    logger.info(f"")
    
    # Boucle infinie pour écouter les messages
    client.loop_forever()
    
except KeyboardInterrupt:
    logger.info(f"\n\n⚠️  Arrêt demandé par l'utilisateur")
except Exception as e:
    logger.error(f"❌ Erreur de connexion: {e}")
    logger.error(f"⚠️  Assurez-vous que:")
    logger.error(f"   1. Mosquitto est installé")
    logger.error(f"   2. Le broker est démarré")
    logger.error(f"   3. L'API FastAPI est en cours d'exécution")
finally:
    client.disconnect()
    logger.info(f"✅ Déconnecté du broker MQTT")
    logger.info(f"="*80)