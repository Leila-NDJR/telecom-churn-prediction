"""
API FastAPI pour la Prédiction du Churn Client
TeleConnect Afrique

Endpoints:
- GET  /health         : Vérifier que l'API fonctionne
- POST /predict        : Prédiction unitaire
- POST /predict_batch  : Prédiction batch (plusieurs clients)
- GET  /model_info     : Informations sur le modèle
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ============================================================================
# FONCTIONS DE FEATURE ENGINEERING
# (Doivent reproduire le Notebook 02)
# ============================================================================

def create_custom_features(df):
    """Crée les 5 features personnalisées (Feature Engineering) :
    AvgMonthlyCharges, TotalServices, ChargesPerService, SeniorWithFamily, TenureCategory
    """
    
    # 1. Feature: TotalServices (Nombre de services actifs)
    service_features = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['TotalServices'] = 0
    # Compter tous les 'Yes'
    for col in service_features:
        df['TotalServices'] += (df[col] == 'Yes').astype(int)
    # Ajouter PhoneService si 'Yes'
    df['TotalServices'] += (df['PhoneService'] == 'Yes').astype(int)
    # Ajouter InternetService si 'DSL' ou 'Fiber optic'
    df['TotalServices'] += (df['InternetService'].isin(['DSL', 'Fiber optic'])).astype(int)


    # 2. Feature: AvgMonthlyCharges (Charges mensuelles moyennes par mois d'ancienneté)
    # Remplacer 0 par 1 pour éviter la division par zéro
    df['AvgMonthlyCharges'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
    # Forcer 0 pour les clients avec tenure=0
    df.loc[df['tenure'] == 0, 'AvgMonthlyCharges'] = 0.0

    # 3. Feature: ChargesPerService (Charges mensuelles moyennes par service)
    # Remplacer 0 par 1 pour éviter la division par zéro dans TotalServices
    df['ChargesPerService'] = df['MonthlyCharges'] / df['TotalServices'].replace(0, 1)
    # Forcer 0 pour les clients sans services
    df.loc[df['TotalServices'] == 0, 'ChargesPerService'] = 0.0
    
    # 4. Feature: SeniorWithFamily (SeniorCitizen=1 AND (Partner=Yes OR Dependents=Yes))
    df['SeniorWithFamily'] = (
        (df['SeniorCitizen'] == 1) & (
            (df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')
        )
    ).astype(int)
    
    # 5. Feature: TenureCategory (Classification de l'ancienneté pour OHE)
    bins = [0, 12, 24, 48, 72]  # <1an, 1-2ans, 2-4ans, 4+ans
    labels = ['< 1 an', '1-2 ans', '2-4 ans', '4+ ans']
    df['TenureCategory'] = pd.cut(
        df['tenure'], 
        bins=bins, 
        labels=labels, 
        right=False, 
        include_lowest=True
    ).astype(str)

    return df

# ============================================================================
# INITIALISATION
# ============================================================================

app = FastAPI(
    title="TeleConnect Churn Prediction API",
    description="API de prédiction du churn client pour TeleConnect Afrique",
    version="1.0.0"
)

# CORS (pour permettre les requêtes depuis un frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle et métadonnées au démarrage
print("🔄 Chargement du modèle...")
try:
    # Chemins absolus pour éviter les erreurs
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    
    model = joblib.load(os.path.join(DATA_DIR, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(DATA_DIR, 'scaler.pkl'))
    
    with open(os.path.join(DATA_DIR, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    with open(os.path.join(DATA_DIR, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    
    # Récupérer le seuil optimisé du metadata, sinon utiliser 0.35 (valeur optimale typique)
    BEST_THRESHOLD = metadata.get('best_threshold', 0.35)
    
    # IMPORTANT: Si tu connais ton seuil optimal du notebook 03, force-le ici:
    # BEST_THRESHOLD = 0.35  # ← Décommente et remplace par ton seuil réel
    
    print("✅ Modèle chargé avec succès !")
    print(f"   Modèle: {metadata['best_model']}")
    print(f"   Seuil optimal: {BEST_THRESHOLD}")
    print(f"   Features: {len(feature_names)}")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    model = None
    metadata = {}


# ============================================================================
# FONCTIONS DE POST-TRAITEMENT ET RECOMMANDATIONS
# (Nécessaires pour les prédictions et la logique métier)
# ============================================================================

def post_process(probability: float) -> tuple[str, str, float]:
    """
    Applique le seuil optimal pour la prédiction et détermine le niveau de risque.
    
    BEST_THRESHOLD est défini à 0.5 dans votre code initial.
    """
    
    # 1. Prédiction binaire
    prediction = "Churn" if probability >= BEST_THRESHOLD else "No Churn"
    
    # 2. Niveau de risque basé sur la probabilité
    if prediction == "Churn":
        if probability >= 0.75:
            risk_level = "High"
            confidence = probability
        elif probability >= BEST_THRESHOLD:
            risk_level = "Medium"
            confidence = probability
        else:
            # Ne devrait pas arriver si prediction == "Churn"
            risk_level = "Medium" 
            confidence = probability
    else:
        # Client No Churn
        risk_level = "Low"
        confidence = 1.0 - probability # Confiance dans le 'No Churn'

    # S'assurer que la confidence est positive
    confidence = abs(confidence)
    
    return prediction, risk_level, confidence


def get_recommendations(probability: float, customer_data: dict) -> list[str]:
    """
    Génère des recommandations personnalisées basées sur la probabilité et les features client.
    """
    recommendations = []
    
    # Recommandations spécifiques si risque élevé
    if probability >= 0.75:
        recommendations.append("Alerte: Client à très haut risque. Contact immédiat du service Rétention.")
    elif probability >= BEST_THRESHOLD:
        recommendations.append("Client à risque modéré. Lancer une campagne de rétention automatisée (email/SMS).")

    # Recommandations basées sur les features (Exemples)
    if customer_data.get('Contract') == 'Month-to-month' and probability >= BEST_THRESHOLD:
        recommendations.append("Proposer une offre de contrat 1 an ou 2 ans avec réduction pour stabiliser.")
    
    if customer_data.get('InternetService') == 'Fiber optic' and customer_data.get('MonthlyCharges', 0) > 90 and probability >= BEST_THRESHOLD:
        recommendations.append("Vérifier la satisfaction des services fibre et envisager un rabais ou un service premium gratuit.")
        
    if customer_data.get('TechSupport') == 'No' and probability >= 0.6:
        recommendations.append("Offrir un mois de support technique gratuit pour améliorer l'expérience client.")

    if not recommendations:
        recommendations.append("Client stable. Suivi standard.")
        
    return recommendations

# ============================================================================
# FIN DES FONCTIONS SUPPLÉMENTAIRES
# ============================================================================

# ============================================================================
# SCHÉMAS PYDANTIC (VALIDATION DES DONNÉES)
# ============================================================================

class CustomerData(BaseModel):
    """Données d'un client pour la prédiction"""
    customerID: Optional[str] = Field(None, description="Identifiant unique du client")

    # Informations démographiques
    gender: str = Field(..., description="Genre: Male ou Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior: 0 ou 1")
    Partner: str = Field(..., description="A un partenaire: Yes ou No")
    Dependents: str = Field(..., description="A des personnes à charge: Yes ou No")
    
    # Informations du compte
    tenure: int = Field(..., ge=0, description="Ancienneté en mois")
    Contract: str = Field(..., description="Type de contrat: Month-to-month, One year, Two year")
    PaperlessBilling: str = Field(..., description="Facturation sans papier: Yes ou No")
    PaymentMethod: str = Field(..., description="Méthode de paiement")
    
    # Informations financières
    MonthlyCharges: float = Field(..., ge=0, description="Charges mensuelles (FCFA)")
    TotalCharges: Optional[float] = Field(None, ge=0, description="Charges totales (FCFA)")
    
    # Services
    PhoneService: str = Field(..., description="Service téléphonique: Yes ou No")
    MultipleLines: str = Field(..., description="Lignes multiples: Yes, No, No phone service")
    InternetService: str = Field(..., description="Service internet: DSL, Fiber optic, No")
    OnlineSecurity: str = Field(..., description="Sécurité en ligne: Yes, No, No internet service")
    OnlineBackup: str = Field(..., description="Backup en ligne: Yes, No, No internet service")
    DeviceProtection: str = Field(..., description="Protection appareil: Yes, No, No internet service")
    TechSupport: str = Field(..., description="Support technique: Yes, No, No internet service")
    StreamingTV: str = Field(..., description="TV streaming: Yes, No, No internet service")
    StreamingMovies: str = Field(..., description="Films streaming: Yes, No, No internet service")
    
    @field_validator('TotalCharges')
    @classmethod
    def impute_total_charges(cls, v, info):
        """Imputer TotalCharges si manquant"""
        if v is None or pd.isna(v):
            tenure = info.data.get('tenure', 0)
            monthly = info.data.get('MonthlyCharges', 0)
            if tenure > 0:
                return monthly * tenure
            return monthly
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "customerid": "ID001",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.50,
                "TotalCharges": 846.00,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes"
            }
        }
    }


class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    customerID: str = Field(..., description="Identifiant unique du client")
    
    churn_probability: float = Field(..., description="Probabilité de churn (0-1)")
    churn_prediction: str = Field(..., description="Prédiction: Churn ou No Churn")
    risk_level: str = Field(..., description="Niveau de risque: Low, Medium, High")
    confidence: float = Field(..., description="Confiance de la prédiction (0-1)")
    threshold_used: float = Field(..., description="Seuil de décision utilisé")
    recommendations: List[str] = Field(..., description="Recommandations d'action")


class BatchPredictionRequest(BaseModel):
    """Requête de prédiction batch"""
    customers: List[CustomerData]


class BatchPredictionResponse(BaseModel):
    """Réponse de prédiction batch"""
    predictions: List[PredictionResponse]
    summary: dict

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def preprocess_customer(customer: CustomerData) -> pd.DataFrame:
    """
    Prétraite les données d'un client pour la prédiction
    """
    # Convertir en dictionnaire
    data = customer.dict()
    
    # Créer un DataFrame
    df = pd.DataFrame([data])
    
    # Feature Engineering (mêmes que dans le preprocessing)
    # 1. AvgMonthlyCharges
    df['AvgMonthlyCharges'] = df.apply(
        lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'],
        axis=1
    )
    
    # 2. TenureCategory
    df['TenureCategory'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-1 an', '1-2 ans', '2-4 ans', '4+ ans']
    )
    
    # 3. TotalServices
    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df['TotalServices'] = 0
    for col in service_cols:
        if col in df.columns:
            df['TotalServices'] += (~df[col].isin(['No', 'No internet service', 'No phone service'])).astype(int)
    
    # 4. ChargesPerService
    df['ChargesPerService'] = df.apply(
        lambda row: row['MonthlyCharges'] / row['TotalServices'] if row['TotalServices'] > 0 else row['MonthlyCharges'],
        axis=1
    )
    
    # 5. SeniorWithFamily
    df['SeniorWithFamily'] = ((df['SeniorCitizen'] == 1) & (df['Partner'] == 'Yes')).astype(int)
    
    # Encodage des variables catégorielles
    # Variables binaires
    binary_map = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1}
    }
    
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # One-Hot Encoding pour les variables multi-catégories
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaymentMethod', 'TenureCategory']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # S'assurer que toutes les features du modèle sont présentes
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Garder seulement les features du modèle dans le bon ordre
    df = df[feature_names]
    
    # Normaliser avec le scaler
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_names
    )
    
    return df_scaled


# ============================================================================
# FONCTION DE RECOMMANDATIONS (Correction de l'accès aux données)
# ============================================================================

def get_recommendations(probability: float, customer_data: dict) -> list[str]:
    """
    Génère des recommandations personnalisées basées sur la probabilité et les features client.
    Note: Utilise .get('key') pour accéder aux éléments du dictionnaire customer_data.
    """
    recommendations = []
    
    # Recommandations spécifiques si risque élevé
    if probability >= 0.75:
        recommendations.append("Alerte: Client à très haut risque. Contact immédiat du service Rétention.")
    elif probability >= BEST_THRESHOLD:
        recommendations.append("Client à risque modéré. Lancer une campagne de rétention automatisée (email/SMS).")

    # Recommandations basées sur les features (Exemples)
    # 🚨 L'accès doit se faire via .get('key') ou ['key']
    if customer_data.get('Contract') == 'Month-to-month' and probability >= BEST_THRESHOLD:
        recommendations.append("Proposer une offre de contrat 1 an ou 2 ans avec réduction pour stabiliser.")
    
    if customer_data.get('InternetService') == 'Fiber optic' and customer_data.get('MonthlyCharges', 0) > 90 and probability >= BEST_THRESHOLD:
        recommendations.append("Vérifier la satisfaction des services fibre et envisager un rabais ou un service premium gratuit.")
        
    if customer_data.get('TechSupport') == 'No' and probability >= 0.6:
        recommendations.append("Offrir un mois de support technique gratuit pour améliorer l'expérience client.")

    if not recommendations:
        recommendations.append("Client stable. Suivi standard.")
        
    return recommendations

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    """Page d'accueil de l'API"""
    return {
        "message": "TeleConnect Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_batch": "/predict_batch (POST)",
            "model_info": "/model_info (GET)",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """Vérifier la santé de l'API"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": metadata.get('best_model', 'Unknown'),
        "threshold": BEST_THRESHOLD,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model_info")
def get_model_info():
    """Informations sur le modèle"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "model_name": metadata.get('best_model', 'Unknown'),
        "threshold": BEST_THRESHOLD,
        "performance": metadata.get('performance', {}),
        "n_features": len(feature_names),
        "top_features": list(metadata.get('feature_importance', {}).get('Feature', {}).values())[:10] if 'feature_importance' in metadata else []
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Prédire le churn pour un client
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Prétraitement
        X = preprocess_customer(customer)
        
        # Prédiction
        probability = model.predict_proba(X)[0][1]
        prediction = "Churn" if probability >= BEST_THRESHOLD else "No Churn"
        
        # Niveau de risque
        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Confiance
        confidence = max(probability, 1 - probability)
        
        # Recommandations
        recommendations = get_recommendations(probability, customer)
        
        return PredictionResponse(
            churn_probability=round(float(probability), 4),
            churn_prediction=prediction,
            risk_level=risk_level,
            confidence=round(float(confidence), 4),
            threshold_used=BEST_THRESHOLD,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


# La classe que vous avez définie pour la requête batch est BatchPredictionRequest
@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest):
    """
    Endpoint pour la prédiction du churn pour un lot de clients.
    """
    try:
        customers_data = batch.customers
        
        # 1. Conversion des données en DataFrame
        customers_df = pd.DataFrame([customer.model_dump() for customer in customers_data])
        
        # 2. Imputation pour TotalCharges manquant (si tenure=0)
        customers_df['TotalCharges'] = customers_df.apply(
            lambda row: 0.0 if row['tenure'] == 0.0 else row['TotalCharges'],
            axis=1
        )
        
        # 3. FEATURE ENGINEERING - Création des nouvelles features (nécessite d'avoir la fonction create_custom_features)
        customers_df = create_custom_features(customers_df) 
        
        # 4. SÉPARER L'ID des features
        customers_features_df = customers_df.drop(columns=['customerID']) 
        
        # ==========================================================
        # 5. ÉTAPE CLÉ MANQUANTE: ONE-HOT ENCODING (OHE)
        # Transforme les colonnes de type 'object' (chaînes) en colonnes binaires (0/1)
        X_ohe = pd.get_dummies(customers_features_df, drop_first=True, dtype=float)
        
        # 6. ÉTAPE CRUCIALE: RE-INDEXING (Synchronisation des colonnes)
        # S'assure que X_ohe a EXACTEMENT les 37 colonnes attendues (feature_names), 
        # dans le bon ordre, et remplace les colonnes manquantes par 0.
        # Cette étape résout l'erreur de "Feature names seen/unseen".
        X_aligned = X_ohe.reindex(columns=feature_names, fill_value=0)
        # ==========================================================
        
        # 7. SCALING
        X_processed = scaler.transform(X_aligned) 

        # ==========================================================
        # 8. EXÉCUTION VECTORISÉE DES PRÉDICTIONS (Optimisation de la Performance)
        # Calcule les probabilités pour tout le batch en une seule fois.
        # Nous prenons la colonne 1 (index 1) qui correspond à la probabilité de CHURN.
        probabilities = model.predict_proba(X_processed)[:, 1]
        # ==========================================================

        # 9. Post-traitement des résultats
        predictions = []
        churn_count = 0
        high_risk_count = 0
        
        # On itère maintenant sur la liste des probabilités vectorisées
        for i, probability in enumerate(probabilities): # <<< On itère sur les résultats, pas sur la matrice d'input
            
            customer_id = customers_data[i].customerID
            
            # Post-traitement (utilise la probabilité calculée)
            prediction, risk_level, confidence = post_process(probability)
            
            if prediction == "Churn":
                churn_count += 1
                if risk_level == "High":
                    high_risk_count += 1
            
            # Les recommandations nécessitent toujours les données client brutes
            recommendations = get_recommendations(probability, customers_df.iloc[i].to_dict())
            
            predictions.append(PredictionResponse(
                customerID=customer_id, 
                churn_probability=round(float(probability), 4),
                churn_prediction=prediction,
                risk_level=risk_level,
                confidence=round(float(confidence), 4),
                threshold_used=BEST_THRESHOLD,
                recommendations=recommendations
            ))
        
        # Résumé
        total = len(predictions)
        summary = {
            "total_customers": total,
            "predicted_churns": churn_count,
            "churn_rate": round(churn_count / total, 4) if total > 0 else 0,
            "high_risk_customers": high_risk_count,
            "medium_risk_customers": sum(1 for p in predictions if p.risk_level == "Medium"),
            "low_risk_customers": sum(1 for p in predictions if p.risk_level == "Low")
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        # Afficher l'erreur détaillée dans les logs du serveur
        print(f"FATAL ERROR IN PREDICT BATCH: {e}", flush=True) 
        # Renvoyer l'erreur 500
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction batch: {str(e)}")


# ============================================================================
# DÉMARRAGE
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)