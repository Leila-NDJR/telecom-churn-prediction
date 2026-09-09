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

    # Récupérer le seuil de décision.
    # PRIORITÉ : le seuil business réellement optimisé (analyse ROI du Notebook 04,
    # data/processed/business_analysis.json -> "optimal_threshold"), qui maximise le
    # revenu net attendu (coût faux positifs + coût faux négatifs + coût des actions
    # de rétention vs revenu sauvé). C'est ce calcul, et non un seuil par défaut
    # arbitraire (0.5), qui doit piloter la décision en production.
    # Repli : model_metadata.json -> "best_threshold", puis 0.35 si rien n'est disponible.
    business_analysis_path = os.path.join(DATA_DIR, 'business_analysis.json')
    business_analysis = {}
    if os.path.exists(business_analysis_path):
        with open(business_analysis_path, 'r') as f:
            business_analysis = json.load(f)

    if 'optimal_threshold' in business_analysis:
        BEST_THRESHOLD = business_analysis['optimal_threshold']
        THRESHOLD_SOURCE = 'business_analysis.json (optimisation ROI)'
    else:
        BEST_THRESHOLD = metadata.get('best_threshold', 0.35)
        THRESHOLD_SOURCE = 'model_metadata.json (repli)'

    print("✅ Modèle chargé avec succès !")
    print(f"   Modèle: {metadata['best_model']}")
    print(f"   Seuil optimal: {BEST_THRESHOLD} (source: {THRESHOLD_SOURCE})")
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

    BEST_THRESHOLD est chargé au démarrage depuis business_analysis.json
    (seuil optimisé business, ~0.30), voir bloc de chargement plus haut.
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


# ----------------------------------------------------------------------------
# Valeur client (CLV proxy) et score de priorisation risque × valeur
# ----------------------------------------------------------------------------
# Durée de vie restante attendue (en mois), estimée à partir de l'ancienneté
# (tenure) MOYENNE RÉELLEMENT OBSERVÉE chez les clients NON churnés du dataset
# d'entraînement (data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv), regroupés par
# type de contrat. Calcul (pandas) :
#   df[df.Churn=='No'].groupby('Contract').tenure.mean()
#     Month-to-month -> 21.03 mois (n=2220)
#     One year       -> 41.67 mois (n=1307)
#     Two year       -> 56.60 mois (n=1647)
# Interprétation : la durée de vie restante d'un client = la durée de vie
# "typique" d'un client fidèle du même type de contrat, moins l'ancienneté déjà
# écoulée, avec un plancher pour ne pas sous-évaluer un client déjà très
# ancien (toujours actif -> encore de la valeur future).
EXPECTED_TENURE_MONTHS_BY_CONTRACT = {
    "Month-to-month": 21.03,
    "One year": 41.67,
    "Two year": 56.60,
}
MIN_REMAINING_TENURE_MONTHS = 6.0
# Horizon par défaut si le type de contrat est absent/inconnu : faute de
# meilleure information dans les données, on retombe sur 12 mois (proxy simple
# "MonthlyCharges x 12").
DEFAULT_REMAINING_TENURE_MONTHS = 12.0

# Segment "haute valeur" : même seuil de facture mensuelle que celui déjà
# utilisé pour l'alerte fibre ci-dessous, réutilisé pour rester cohérent.
HIGH_VALUE_MONTHLY_CHARGES = 90.0


def estimate_remaining_tenure_months(contract: Optional[str], tenure) -> float:
    """Estime la durée de vie restante (mois) d'un client à partir de son
    contrat et de son ancienneté actuelle (voir constantes ci-dessus)."""
    try:
        tenure = float(tenure)
    except (TypeError, ValueError):
        tenure = 0.0

    expected_lifetime = EXPECTED_TENURE_MONTHS_BY_CONTRACT.get(contract)
    if expected_lifetime is None:
        return DEFAULT_REMAINING_TENURE_MONTHS

    return max(expected_lifetime - tenure, MIN_REMAINING_TENURE_MONTHS)


def calculate_clv_proxy(customer_data: dict) -> float:
    """CLV proxy = MonthlyCharges x durée de vie restante estimée (mois).
    C'est une approximation simple de la valeur future du client, utilisée
    uniquement pour prioriser les contacts de rétention (pas une vraie
    valorisation financière du client)."""
    monthly_charges = customer_data.get('MonthlyCharges', 0) or 0
    try:
        monthly_charges = float(monthly_charges)
    except (TypeError, ValueError):
        monthly_charges = 0.0

    remaining_months = estimate_remaining_tenure_months(
        customer_data.get('Contract'), customer_data.get('tenure', 0)
    )
    return round(monthly_charges * remaining_months, 2)


def calculate_priority_score(probability: float, clv_proxy: float) -> float:
    """Score de priorisation = probabilité de churn x valeur client (CLV proxy).
    Sert à classer les clients à contacter en premier quand le budget de
    rétention (temps agent, coût des offres) est limité : à risque égal, on
    traite d'abord les clients à forte valeur ; à valeur égale, on traite
    d'abord les clients les plus à risque."""
    return round(float(probability) * clv_proxy, 2)


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


class Recommendation(BaseModel):
    """Recommandation actionnable pour l'équipe terrain"""
    rule: str = Field(..., description="Règle métier déclenchée")
    priority: int = Field(..., description="Ordre de priorité de l'action pour ce client (1 = plus urgent)")
    channel: str = Field(..., description="Canal de contact suggéré (SMS, Email, Appel agent rétention, ...)")
    offer: str = Field(..., description="Offre indicative chiffrée (remise en % et FCFA/mois estimés)")


class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    customerID: str = Field(..., description="Identifiant unique du client")

    churn_probability: float = Field(..., description="Probabilité de churn (0-1)")
    churn_prediction: str = Field(..., description="Prédiction: Churn ou No Churn")
    risk_level: str = Field(..., description="Niveau de risque: Low, Medium, High")
    confidence: float = Field(..., description="Confiance de la prédiction (0-1)")
    threshold_used: float = Field(..., description="Seuil de décision utilisé")
    clv_proxy: float = Field(..., description="Valeur client estimée en FCFA (CLV proxy) = MonthlyCharges x durée de vie restante estimée")
    priority_score: float = Field(..., description="Score de priorisation = churn_probability x clv_proxy. Classe les clients à contacter en premier avec un budget de rétention limité")
    recommendations: List[Recommendation] = Field(..., description="Recommandations d'action")


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
# FONCTION DE RECOMMANDATIONS (actionnable : canal, priorité, offre chiffrée)
# ============================================================================

def get_recommendations(probability: float, customer_data: dict) -> List[dict]:
    """
    Génère des recommandations personnalisées et actionnables pour l'équipe
    terrain. Chaque recommandation précise :
      - rule     : la règle métier déclenchée
      - priority : ordre de priorité de l'action pour CE client (1 = plus urgent)
      - channel  : canal de contact suggéré (SMS pour un contrat mensuel afin
                   de toucher vite un client peu engagé, appel agent dédié
                   pour les clients à haute valeur qui justifient un contact
                   humain, email/SMS pour le reste)
      - offer    : offre indicative chiffrée (remise en % et FCFA/mois estimés
                   à partir de la facture mensuelle réelle du client)

    Le classement global "qui contacter en premier" (au-delà d'un seul client)
    doit se faire sur le champ `priority_score` de la réponse API (risque x
    valeur client), pas sur ces priorités locales par règle.

    Note: utilise .get('key') pour accéder aux éléments du dictionnaire
    customer_data (qui peut être un customer.model_dump() ou une ligne de
    DataFrame convertie en dict).
    """
    recommendations = []

    contract = customer_data.get('Contract')
    monthly_charges = customer_data.get('MonthlyCharges', 0) or 0
    try:
        monthly_charges = float(monthly_charges)
    except (TypeError, ValueError):
        monthly_charges = 0.0
    high_value = monthly_charges > HIGH_VALUE_MONTHLY_CHARGES

    # Règle 1 : risque très élevé -> contact immédiat, canal humain pour les
    # clients à forte valeur, centre d'appels standard sinon.
    if probability >= 0.75:
        recommendations.append({
            "rule": "Risque très élevé de churn",
            "priority": 1,
            "channel": "Appel agent rétention dédié" if high_value else "Appel centre d'appels",
            "offer": (
                f"Remise de 20% sur 3 mois (~{round(monthly_charges * 0.20, 2)} FCFA/mois)"
                + (" + 1 mois de service premium offert" if high_value else "")
            ),
        })
    # Règle 2 : risque modéré -> campagne de rétention.
    elif probability >= BEST_THRESHOLD:
        recommendations.append({
            "rule": "Risque modéré de churn",
            "priority": 2,
            "channel": (
                "Appel agent rétention" if high_value
                else ("SMS" if contract == "Month-to-month" else "Email")
            ),
            "offer": f"Remise de 10% sur 2 mois (~{round(monthly_charges * 0.10, 2)} FCFA/mois)",
        })

    # Règle 3 : contrat mensuel à risque -> proposer un engagement plus long
    # (le SMS est privilégié car c'est un client peu engagé, à toucher vite).
    if contract == 'Month-to-month' and probability >= BEST_THRESHOLD:
        recommendations.append({
            "rule": "Contrat mensuel instable",
            "priority": 1 if high_value else 3,
            "channel": "SMS",
            "offer": (
                "Remise de 15% sur 12 mois en cas de passage à un contrat 1 an, "
                "ou 25% sur 24 mois pour un contrat 2 ans "
                f"(~{round(monthly_charges * 0.15, 2)} FCFA/mois d'économie sur l'offre 1 an)"
            ),
        })

    # Règle 4 : fibre chère -> vérifier la satisfaction du service, contact humain.
    if contract is not None and customer_data.get('InternetService') == 'Fiber optic' \
            and monthly_charges > HIGH_VALUE_MONTHLY_CHARGES and probability >= BEST_THRESHOLD:
        recommendations.append({
            "rule": "Client fibre à forte facture",
            "priority": 1,
            "channel": "Appel agent rétention",
            "offer": f"Rabais de 15% sur la facture fibre (~{round(monthly_charges * 0.15, 2)} FCFA/mois) ou upgrade gratuit du service",
        })

    # Règle 5 : pas de support technique -> geste de service, canal léger.
    if customer_data.get('TechSupport') == 'No' and probability >= 0.6:
        recommendations.append({
            "rule": "Absence de support technique",
            "priority": 3,
            "channel": "Email",
            "offer": "1 mois de support technique offert",
        })

    if not recommendations:
        recommendations.append({
            "rule": "Client stable",
            "priority": 5,
            "channel": "Aucun contact proactif",
            "offer": "Suivi standard, pas d'offre",
        })

    recommendations.sort(key=lambda r: r["priority"])
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

        # Post-traitement (prédiction + niveau de risque), cohérent avec /predict_batch :
        # utilise le même BEST_THRESHOLD partout, au lieu d'un seuil de risque
        # (0.7/0.4) déconnecté du seuil de décision réellement appliqué.
        prediction, risk_level, confidence = post_process(probability)

        # Valeur client (CLV proxy) et score de priorisation risque x valeur
        customer_dict = customer.model_dump()
        clv_proxy = calculate_clv_proxy(customer_dict)
        priority_score = calculate_priority_score(probability, clv_proxy)

        # Recommandations (nécessite un dict, pas l'objet pydantic)
        recommendations = get_recommendations(probability, customer_dict)

        return PredictionResponse(
            customerID=customer.customerID or "N/A",
            churn_probability=round(float(probability), 4),
            churn_prediction=prediction,
            risk_level=risk_level,
            confidence=round(float(confidence), 4),
            threshold_used=BEST_THRESHOLD,
            clv_proxy=clv_proxy,
            priority_score=priority_score,
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
            
            # Les recommandations et le CLV nécessitent toujours les données client brutes
            customer_dict = customers_df.iloc[i].to_dict()
            recommendations = get_recommendations(probability, customer_dict)
            clv_proxy = calculate_clv_proxy(customer_dict)
            priority_score = calculate_priority_score(probability, clv_proxy)

            predictions.append(PredictionResponse(
                customerID=customer_id,
                churn_probability=round(float(probability), 4),
                churn_prediction=prediction,
                risk_level=risk_level,
                confidence=round(float(confidence), 4),
                threshold_used=BEST_THRESHOLD,
                clv_proxy=clv_proxy,
                priority_score=priority_score,
                recommendations=recommendations
            ))

        # Trier par priority_score décroissant : avec un budget de rétention
        # limité, l'équipe terrain traite la liste dans l'ordre renvoyé.
        predictions.sort(key=lambda p: p.priority_score, reverse=True)

        # Résumé
        total = len(predictions)
        summary = {
            "total_customers": total,
            "predicted_churns": churn_count,
            "churn_rate": round(churn_count / total, 4) if total > 0 else 0,
            "high_risk_customers": high_risk_count,
            "medium_risk_customers": sum(1 for p in predictions if p.risk_level == "Medium"),
            "low_risk_customers": sum(1 for p in predictions if p.risk_level == "Low"),
            "avg_priority_score": round(sum(p.priority_score for p in predictions) / total, 2) if total > 0 else 0,
            "top_priority_customer_ids": [p.customerID for p in predictions[:5]]
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