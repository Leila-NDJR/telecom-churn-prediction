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
# SCHÉMAS PYDANTIC (VALIDATION DES DONNÉES)
# ============================================================================

class CustomerData(BaseModel):
    """Données d'un client pour la prédiction"""
    
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


def get_recommendations(probability: float, customer_data: CustomerData) -> List[str]:
    """
    Génère des recommandations basées sur la probabilité de churn
    """
    recs = []
    
    if probability > 0.7:
        recs.append("🚨 PRIORITÉ ÉLEVÉE: Contacter immédiatement")
        recs.append("💰 Proposer une offre de rétention personnalisée")
        recs.append("🎁 Réduction de 20-30% pendant 6 mois")
    elif probability > 0.5:
        recs.append("⚠️ Risque modéré: Planifier un contact dans les 2 semaines")
        recs.append("📞 Appel de satisfaction client")
        recs.append("🎁 Offre de fidélité ou upgrade service")
    else:
        recs.append("✅ Risque faible: Maintenir la relation")
        recs.append("📧 Campagne d'engagement standard")
    
    # Recommandations basées sur les features
    if customer_data.Contract == "Month-to-month":
        recs.append("📝 Proposer un contrat annuel avec avantages")
    
    if customer_data.tenure < 12:
        recs.append("🎯 Client récent: Programme de bienvenue spécial")
    
    if customer_data.InternetService == "Fiber optic" and customer_data.MonthlyCharges > 80:
        recs.append("💸 Client à forte valeur: Offre VIP personnalisée")
    
    total_services = sum([
        customer_data.PhoneService == "Yes",
        customer_data.InternetService != "No",
        customer_data.OnlineSecurity == "Yes",
        customer_data.OnlineBackup == "Yes",
        customer_data.DeviceProtection == "Yes",
        customer_data.TechSupport == "Yes",
        customer_data.StreamingTV == "Yes",
        customer_data.StreamingMovies == "Yes"
    ])
    
    if total_services < 3:
        recs.append("📦 Proposer un bundle de services attractif")
    
    return recs

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


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Prédire le churn pour plusieurs clients
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        predictions = []
        churn_count = 0
        high_risk_count = 0
        
        for customer in request.customers:
            # Prétraitement
            X = preprocess_customer(customer)
            
            # Prédiction
            probability = model.predict_proba(X)[0][1]
            prediction = "Churn" if probability >= BEST_THRESHOLD else "No Churn"
            
            if prediction == "Churn":
                churn_count += 1
            
            # Niveau de risque
            if probability >= 0.7:
                risk_level = "High"
                high_risk_count += 1
            elif probability >= 0.4:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            confidence = max(probability, 1 - probability)
            recommendations = get_recommendations(probability, customer)
            
            predictions.append(PredictionResponse(
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
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction batch: {str(e)}")


# ============================================================================
# DÉMARRAGE
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)