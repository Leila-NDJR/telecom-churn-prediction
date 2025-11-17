import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_NAME = "customers.db"

# Définir la structure de la table avec toutes les colonnes du modèle
CREATE_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS customers (
    customerID TEXT PRIMARY KEY,
    gender TEXT,
    SeniorCitizen INTEGER,
    Partner TEXT,
    Dependents TEXT,
    tenure INTEGER,
    Contract TEXT,
    PaperlessBilling TEXT,
    PaymentMethod TEXT,
    MonthlyCharges REAL,
    TotalCharges REAL,
    PhoneService TEXT,
    MultipleLines TEXT,
    InternetService TEXT,
    OnlineSecurity TEXT,
    OnlineBackup TEXT,
    DeviceProtection TEXT,
    TechSupport TEXT,
    StreamingTV TEXT,
    StreamingMovies TEXT
);
"""

# 10 clients fictifs avec les variables complètes (ajout d'un ID unique)
FAKE_CUSTOMERS = [
    ('ID001', 'Female', 0, 'Yes', 'No', 6, 'Month-to-month', 'Yes', 'Electronic check', 85.50, 513.00, 'Yes', 'No', 'Fiber optic', 'No', 'No', 'No', 'No', 'Yes', 'Yes'),
    ('ID002', 'Male', 1, 'No', 'No', 48, 'Two year', 'No', 'Bank transfer', 55.20, 2650.00, 'Yes', 'Yes', 'DSL', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No'),
    ('ID003', 'Female', 0, 'Yes', 'Yes', 24, 'One year', 'Yes', 'Credit card', 70.00, 1680.00, 'Yes', 'Yes', 'Fiber optic', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes'),
    ('ID004', 'Male', 0, 'No', 'No', 12, 'Month-to-month', 'Yes', 'Mailed check', 45.00, 540.00, 'No', 'No phone service', 'DSL', 'No', 'No', 'No', 'No', 'No', 'No'),
    ('ID005', 'Female', 1, 'Yes', 'No', 72, 'Two year', 'Yes', 'Credit card', 110.50, 7956.65, 'Yes', 'Yes', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'),
    ('ID006', 'Male', 0, 'No', 'No', 1, 'Month-to-month', 'No', 'Electronic check', 20.00, 20.00, 'Yes', 'No', 'No', 'No internet service', 'No internet service', 'No internet service', 'No internet service', 'No internet service', 'No internet service'),
    ('ID007', 'Female', 0, 'Yes', 'Yes', 36, 'One year', 'No', 'Bank transfer', 60.50, 2178.00, 'Yes', 'Yes', 'DSL', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'),
    ('ID008', 'Male', 0, 'No', 'No', 5, 'Month-to-month', 'Yes', 'Electronic check', 90.00, 450.00, 'Yes', 'Yes', 'Fiber optic', 'No', 'No', 'Yes', 'No', 'No', 'No'),
    ('ID009', 'Female', 1, 'Yes', 'No', 18, 'Month-to-month', 'Yes', 'Credit card', 80.20, 1443.60, 'Yes', 'No', 'Fiber optic', 'No', 'Yes', 'No', 'No', 'No', 'Yes'),
    ('ID010', 'Male', 0, 'No', 'No', 60, 'Two year', 'No', 'Mailed check', 25.50, 1530.00, 'Yes', 'Yes', 'No', 'No internet service', 'No internet service', 'No internet service', 'No internet service', 'No internet service', 'No internet service')
]

def setup_database():
    try:
        logger.info(f"🔄 Connexion à la base de données '{DB_NAME}'...")
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        logger.info("🔧 Création de la table 'customers' (si elle n'existe pas)...")
        cursor.execute(CREATE_TABLE_QUERY)
        
        logger.info(f"➕ Insertion de {len(FAKE_CUSTOMERS)} clients fictifs...")
        
        # Requête pour insérer les données
        insert_query = f"""
        INSERT OR IGNORE INTO customers VALUES (
            {', '.join(['?' for _ in range(len(FAKE_CUSTOMERS[0]))])}
        )
        """
        
        cursor.executemany(insert_query, FAKE_CUSTOMERS)
        
        conn.commit()
        logger.info(f"✅ Base de données '{DB_NAME}' initialisée avec succès !")
        
    except sqlite3.Error as e:
        logger.error(f"❌ Erreur SQLite: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    setup_database()