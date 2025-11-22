import sqlite3
import logging
import random

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_NAME = "customers.db"
NUM_CUSTOMERS = 40 # On va générer 40 clients au total

def generate_loyal_customer(index):
    """Génère un profil de client fidèle (No Churn)"""
    return (
        f'LOYAL-{index:03d}',     # customerID
        random.choice(['Male', 'Female']), # gender
        0,                        # SeniorCitizen (Non)
        'Yes',                    # Partner (Oui)
        'Yes',                    # Dependents (Oui)
        random.randint(48, 72),   # tenure (Longue: 4 à 6 ans)
        'Two year',               # Contract (Engagement 2 ans)
        'No',                     # PaperlessBilling
        'Credit card (automatic)',# PaymentMethod
        round(random.uniform(20, 60), 2), # MonthlyCharges (Faibles/Moyennes)
        0.0, # TotalCharges (Calculé après)
        'Yes',                    # PhoneService
        'No',                     # MultipleLines
        'DSL',                    # InternetService (Pas de fibre)
        'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No' # Services de sécurité activés
    )

def generate_risky_customer(index):
    """Génère un profil de client à risque (Churn)"""
    return (
        f'RISK-{index:03d}',      # customerID
        random.choice(['Male', 'Female']),
        random.choice([0, 1]),
        'No',                     # Partner (Non)
        'No',                     # Dependents (Non)
        random.randint(1, 12),    # tenure (Courte: < 1 an)
        'Month-to-month',         # Contract (Mensuel)
        'Yes',                    # PaperlessBilling
        'Electronic check',       # PaymentMethod
        round(random.uniform(70, 110), 2), # MonthlyCharges (Élevées)
        0.0,
        'Yes', 
        'Yes',
        'Fiber optic',            # InternetService (Fibre, souvent source de problèmes)
        'No', 'No', 'No', 'No', 'Yes', 'Yes' # Pas de services de sécurité, mais du streaming
    )

def insert_new_customers():
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        logger.info(f"🔄 Génération de {NUM_CUSTOMERS} clients (Mixtes)...")
        
        customers_data = []
        
        # Générer 50% de fidèles et 50% de risqués
        for i in range(1, (NUM_CUSTOMERS // 2) + 1):
            # 1. Client Fidèle
            loyal = list(generate_loyal_customer(i))
            # Calcul correct du TotalCharges
            loyal[10] = round(loyal[9] * loyal[5], 2)
            customers_data.append(tuple(loyal))
            
            # 2. Client à Risque
            risky = list(generate_risky_customer(i))
            # Calcul correct du TotalCharges
            risky[10] = round(risky[9] * risky[5], 2)
            customers_data.append(tuple(risky))

        # Requête d'insertion
        insert_query = """
        INSERT OR IGNORE INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.executemany(insert_query, customers_data)
        conn.commit()
        logger.info(f"✅ {len(customers_data)} clients insérés (Mélange Fidèles/Risqués) dans '{DB_NAME}'")
        
    except sqlite3.Error as e:
        logger.error(f"❌ Erreur SQLite lors de l'insertion: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    insert_new_customers()