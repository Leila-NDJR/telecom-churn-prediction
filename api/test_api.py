import requests
import json
import time 

BASE_URL = "http://localhost:8000"

print("="*80)
print("TEST DE L'API TELECONNECT CHURN PREDICTION")
print("="*80)

# ============================================================================
# Données de test pour le batch (30 clients pour mesurer l'overhead)
# ============================================================================
# Clients de base pour duplication
BASE_CUSTOMERS = [
    {
        "customerID": "B-001", "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 2, "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.00, "TotalCharges": 190.00,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes"
    },
    {
        "customerID": "B-002", "gender": "Male", "SeniorCitizen": 1, "Partner": "Yes",
        "Dependents": "No", "tenure": 60, "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 20.00, "TotalCharges": 1200.00,
        "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "No",
        "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service", "TechSupport": "No internet service",
        "StreamingTV": "No internet service", "StreamingMovies": "No internet service"
    },
    {
        "customerID": "B-003", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "Yes", "tenure": 1, "Contract": "Month-to-month", "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 19.90, "TotalCharges": 19.90,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "No",
        "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service", "TechSupport": "No internet service",
        "StreamingTV": "No internet service", "StreamingMovies": "No internet service"
    }
]

# Créer un lot de 30 clients en dupliquant les données de base
BATCH_TEST_DATA = []
for i in range(1, 31): 
    # Utilise le modulo pour cycler entre les 3 clients de base
    customer = BASE_CUSTOMERS[(i-1) % len(BASE_CUSTOMERS)].copy()
    customer["customerID"] = f"BatchTest-{i:02d}"
    BATCH_TEST_DATA.append(customer)
# ============================================================================


# Test 1 : Health Check
print("\n🏥 Test 1 : Health Check")
print("-"*80)
response = requests.get(f"{BASE_URL}/health")
print(json.dumps(response.json(), indent=2))

# Test 2 : Model Info
print("\n📊 Test 2 : Informations du Modèle")
print("-"*80)
response = requests.get(f"{BASE_URL}/model_info")
print(json.dumps(response.json(), indent=2))

# Test 3 : Prédiction Batch (30 clients) et Mesure de Latence
print("\n🚀 Test 3 : Prédiction Batch (30 clients) et MESURE DE LATENCE")
print("-"*80)

# Préparation du payload
payload = {"customers": BATCH_TEST_DATA}

# Début de la mesure
start_time = time.time()
response = requests.post(f"{BASE_URL}/predict_batch", json=payload)
end_time = time.time()
latency_ms = (end_time - start_time) * 1000 # Temps en millisecondes

# Vérification du statut
if response.status_code == 200:
    result = response.json()
    
    # Affichage de la Latence
    status_emoji = "✅" if latency_ms < 200 else "⚠️"
    print(f"\n{status_emoji} TEMPS DE RÉPONSE API: {latency_ms:.2f} ms (< 200ms requis)")
    
    # Affichage du résumé
    print("\n📊 Résumé des Prédictions :")
    print(json.dumps(result['summary'], indent=2))
    
    # Affichage du premier client pour vérification
    print("\n🔍 Détail du premier client (BatchTest-01) :")
    print(json.dumps(result['predictions'][0], indent=2))
    
else:
    print(f"❌ Échec de la prédiction batch. Statut: {response.status_code}")
    try:
        print(f"Erreur détaillée: {response.json()}")
    except:
        print(f"Réponse: {response.text}")
        
print("\n" + "="*80)