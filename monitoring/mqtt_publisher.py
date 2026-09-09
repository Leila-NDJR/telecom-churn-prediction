"""
MQTT Publisher - Envoie des demandes de prédictions batch via MQTT
TeleConnect Afrique
"""

import os
import paho.mqtt.client as mqtt
import json
import time
import sqlite3 # NOUVEAU: Importation de la bibliothèque SQLite
from dotenv import load_dotenv

load_dotenv()

# Configuration MQTT
MQTT_BROKER = "localhost"  # Adresse du broker Mosquitto
MQTT_PORT = 1883
MQTT_TOPIC_REQUEST = "teleconnect/churn/prediction/request"
MQTT_TOPIC_RESPONSE = "teleconnect/churn/prediction/response"

# === Identifiants MQTT (définis dans .env, jamais commités) ===
MQTT_USER = os.environ["MQTT_USER"]
MQTT_PASSWORD = os.environ["MQTT_PASSWORD"]
# ===============================

# Configuration SQLite
DB_NAME = "customers.db" # NOUVEAU: Nom du fichier de la BDD

print("="*80)
print("MQTT PUBLISHER - TeleConnect Churn Prediction")
print("="*80)

# Fonction de lecture de la BDD
def fetch_customers_from_db():
    """Récupère tous les clients de la base de données SQLite."""
    customers_list = []
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        # Permet de récupérer les colonnes sous forme de dictionnaire/clé
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        
        print(f"🗄️  Lecture des clients depuis '{DB_NAME}'...")
        cursor.execute("SELECT * FROM customers")
        rows = cursor.fetchall()
        
        # Convertir les objets 'Row' en dictionnaires standards
        customers_list = [dict(row) for row in rows]
        
        print(f"✅ {len(customers_list)} clients récupérés.")
        
    except sqlite3.Error as e:
        print(f"❌ Erreur SQLite lors de la lecture: {e}")
    finally:
        if conn:
            conn.close()
    
    return customers_list

# Callback lors de la connexion
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✅ Connecté au broker MQTT ({MQTT_BROKER}:{MQTT_PORT})")
        print(f"📡 Topic requête: {MQTT_TOPIC_REQUEST}")
        print(f"📡 Topic réponse: {MQTT_TOPIC_RESPONSE}")
        # S'abonner au topic de réponse
        client.subscribe(MQTT_TOPIC_RESPONSE)
        print(f"✅ Abonné au topic de réponse")
    else:
        print(f"❌ Échec de connexion, code: {rc}")

# Callback lors de la réception d'un message
def on_message(client, userdata, msg):
    print(f"\n📩 RÉPONSE REÇUE sur {msg.topic}")
    print("-"*80)
    try:
        response = json.loads(msg.payload.decode())
        
        # Afficher le résumé
        summary = response.get('summary', {})
        print(f"📊 RÉSUMÉ:")
        print(f"   Clients analysés: {summary.get('total_customers', 0)}")
        print(f"   Churns prédits: {summary.get('predicted_churns', 0)}")
        print(f"   Taux de churn: {summary.get('churn_rate', 0)*100:.1f}%")
        print(f"   Risque élevé: {summary.get('high_risk_customers', 0)}")
        print(f"   Risque moyen: {summary.get('medium_risk_customers', 0)}")
        print(f"   Risque faible: {summary.get('low_risk_customers', 0)}")
        
        # Afficher le détail des prédictions (pour tous les clients)
        predictions = response.get('predictions', [])
        print(f"\n📋 DÉTAIL ({len(predictions)} clients analysés):")
        
        for pred in predictions:
            # Récupérer l'ID du client, ou afficher 'N/A' si absent
            client_id = pred.get('customerID', 'ID non trouvé') 
            
            print(f"   ID Client {client_id} (Churn: {pred['churn_prediction']}) "
                  f"Probabilité: {pred['churn_probability']*100:.1f}% | Niveau de risque: {pred['risk_level']}")
        print("-"*80)
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement de la réponse: {e}")

# Créer le client MQTT
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "TeleConnectPublisher")

# Ajouter l'authentification
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

client.on_connect = on_connect
client.on_message = on_message

# Connexion au broker
print(f"\n🔄 Connexion au broker MQTT...")
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()  # Démarrer la boucle en arrière-plan
    time.sleep(2)  # Attendre la connexion
except Exception as e:
    print(f"❌ Erreur de connexion: {e}")
    print(f"⚠️  Assurez-vous que Mosquitto est installé et démarré!")
    exit(1)

# Préparer les données clients pour batch (Lecture depuis la BDD)
print(f"\n📦 Préparation du batch de clients...")

# Remplacement de la liste customers_batch hardcodée
customers_batch = fetch_customers_from_db() 

if not customers_batch:
    print("⚠️  Arrêt: Aucune donnée à publier.")
    exit(0)

# Publier la requête
message = {
    "timestamp": time.time(),
    "customers": customers_batch
}

print(f"📤 Envoi de {len(customers_batch)} clients pour prédiction...")
print(f"🕐 Timestamp: {message['timestamp']}")

client.publish(MQTT_TOPIC_REQUEST, json.dumps(message))
print(f"✅ Message envoyé sur {MQTT_TOPIC_REQUEST}")

print(f"\n⏳ En attente de la réponse...")
print(f"   (Le subscriber doit être en cours d'exécution)")

# Garder le script actif pour recevoir la réponse
try:
    time.sleep(30)  # Attendre 30 secondes
except KeyboardInterrupt:
    print(f"\n\n⚠️  Interruption par l'utilisateur")

# Déconnexion
client.loop_stop()
client.disconnect()
print(f"\n✅ Déconnecté du broker MQTT")
print(f"="*80)