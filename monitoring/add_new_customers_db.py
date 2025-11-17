import sqlite3
import logging
import random
import uuid # Pour générer des ID clients uniques

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_NAME = "customers.db"
NUM_NEW_CUSTOMERS = 30

# Définition des options pour la génération de données aléatoires
OPTIONS = {
    "gender": ['Female', 'Male'],
    "SeniorCitizen": [0, 1],
    "Partner": ['Yes', 'No'],
    "Dependents": ['Yes', 'No'],
    "Contract": ['Month-to-month', 'One year', 'Two year'],
    "PaperlessBilling": ['Yes', 'No'],
    "PaymentMethod": ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    "PhoneService": ['Yes', 'No'],
    "MultipleLines": ['Yes', 'No', 'No phone service'],
    "InternetService": ['DSL', 'Fiber optic', 'No'],
    "SecurityFeatures": ['Yes', 'No', 'No internet service'] # Utilisé pour plusieurs colonnes
}

def generate_customer_data(customer_number):
    """Génère une ligne complète de données client fictives."""
    
    # 1. Générer les variables de base
    data = {}
    data['customerID'] = f"ID{1000 + customer_number}" # ID unique
    data['gender'] = random.choice(OPTIONS["gender"])
    data['SeniorCitizen'] = random.choice(OPTIONS["SeniorCitizen"])
    data['Partner'] = random.choice(OPTIONS["Partner"])
    data['Dependents'] = random.choice(OPTIONS["Dependents"])
    data['Contract'] = random.choice(OPTIONS["Contract"])
    data['PaperlessBilling'] = random.choice(OPTIONS["PaperlessBilling"])
    data['PaymentMethod'] = random.choice(OPTIONS["PaymentMethod"])
    data['PhoneService'] = random.choice(OPTIONS["PhoneService"])
    data['InternetService'] = random.choice(OPTIONS["InternetService"])

    # 2. Tenure (durée d'engagement)
    # Plus de nouveaux clients (tenure basse) et de très fidèles (tenure haute)
    data['tenure'] = random.choice([1, 2, 3, 6, 12, 24, 36, 48, 60, 72])
    
    # 3. MonthlyCharges et TotalCharges
    if data['InternetService'] == 'No':
        # Faibles charges pour les clients sans internet
        data['MonthlyCharges'] = round(random.uniform(18.25, 25.50), 2)
    elif data['InternetService'] == 'DSL':
        data['MonthlyCharges'] = round(random.uniform(30.00, 70.00), 2)
    else: # Fiber optic
        data['MonthlyCharges'] = round(random.uniform(70.00, 118.75), 2)
        
    data['TotalCharges'] = round(data['MonthlyCharges'] * data['tenure'] + random.uniform(0, 50), 2)
    
    # 4. Dépendance des services internet/téléphone
    if data['PhoneService'] == 'No':
        data['MultipleLines'] = 'No phone service'
    else:
        data['MultipleLines'] = random.choice(['Yes', 'No'])

    if data['InternetService'] == 'No':
        # Si pas d'internet, tous les services internet sont 'No internet service'
        default_internet_service = 'No internet service'
        data['OnlineSecurity'] = default_internet_service
        data['OnlineBackup'] = default_internet_service
        data['DeviceProtection'] = default_internet_service
        data['TechSupport'] = default_internet_service
        data['StreamingTV'] = default_internet_service
        data['StreamingMovies'] = default_internet_service
    else:
        # Services aléatoires pour les clients avec internet
        data['OnlineSecurity'] = random.choice(OPTIONS["SecurityFeatures"])
        data['OnlineBackup'] = random.choice(OPTIONS["SecurityFeatures"])
        data['DeviceProtection'] = random.choice(OPTIONS["SecurityFeatures"])
        data['TechSupport'] = random.choice(OPTIONS["SecurityFeatures"])
        data['StreamingTV'] = random.choice(OPTIONS["SecurityFeatures"])
        data['StreamingMovies'] = random.choice(OPTIONS["SecurityFeatures"])
        
    # 5. Créer le tuple dans le bon ordre (20 colonnes)
    customer_tuple = (
        data['customerID'], data['gender'], data['SeniorCitizen'], data['Partner'],
        data['Dependents'], data['tenure'], data['Contract'], data['PaperlessBilling'],
        data['PaymentMethod'], data['MonthlyCharges'], data['TotalCharges'],
        data['PhoneService'], data['MultipleLines'], data['InternetService'],
        data['OnlineSecurity'], data['OnlineBackup'], data['DeviceProtection'],
        data['TechSupport'], data['StreamingTV'], data['StreamingMovies']
    )
    return customer_tuple

def insert_new_customers():
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        logger.info(f"🔄 Génération de {NUM_NEW_CUSTOMERS} clients fictifs...")
        new_customers_data = [generate_customer_data(i) for i in range(1, NUM_NEW_CUSTOMERS + 1)]
        
        # Requête pour insérer (20 "?" pour les 20 colonnes)
        insert_query = """
        INSERT OR IGNORE INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.executemany(insert_query, new_customers_data)
        conn.commit()
        logger.info(f"✅ {cursor.rowcount} nouveaux clients insérés dans '{DB_NAME}'")
        
    except sqlite3.Error as e:
        logger.error(f"❌ Erreur SQLite lors de l'insertion: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    insert_new_customers()