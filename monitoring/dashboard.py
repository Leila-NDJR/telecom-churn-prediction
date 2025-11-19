import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime

# Configuration de la BDD
DB_NAME = "customers.db"
TABLE_NAME = "predictions_log"

# --- CONFIGURATION DE LA PAGE STREAMLIT ---
st.set_page_config(
    page_title="Dashboard Churn TeleConnect",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FONCTION POUR CHARGER LES DONNÉES ---
@st.cache_data(ttl=60) # Actualise les données toutes les 60 secondes
def load_data():
    """Charge toutes les prédictions de la base de données."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        # Trier par date de prédiction pour avoir les plus récentes en haut
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY prediction_date DESC", conn)
        return df
    except Exception as e:
        # En cas d'erreur (ex: fichier DB non trouvé), retourne un DataFrame vide
        st.error(f"Erreur de connexion à la BDD ou de lecture : {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

# --- DÉBUT DU DASHBOARD ---
st.title("📊 Tableau de Bord de Prédiction du Churn")
st.markdown("Suivi en temps réel des prédictions de désabonnement client pour TeleConnect Afrique.")

df_predictions = load_data()

if df_predictions.empty:
    st.warning("Aucune donnée de prédiction trouvée dans la base de données. Lancez le Publisher pour générer des données.")
else:
    
    # 1. PRÉPARATION DES DONNÉES
    
    # A. Identifier le dernier batch
    latest_timestamp = df_predictions['prediction_date'].max()
    
    # B. Filtrer les données pour ne garder que ce dernier batch (pour les métriques principales)
    df_latest = df_predictions[df_predictions['prediction_date'] == latest_timestamp]
    
    # C. Définir le DataFrame à afficher dans le tableau (le dernier batch est le plus pertinent)
    df_display_table = df_latest.copy()

    # --- EN-TÊTE : STATISTIQUES GLOBALEs ET FILTRES ---
    
    # Affichage du total historique pour contexte
    st.subheader(f"Historique total : {len(df_predictions)} prédictions enregistrées")
    st.markdown("---")

    st.subheader(f"Statistiques du Dernier Batch ({len(df_latest)} clients)")

    # 2. CALCUL DES MÉTRIQUES CLÉS (sur le dernier batch)
    
    total_customers = len(df_latest)
    
    # S'assurer que 'Churn' est bien le terme utilisé par l'API
    churn_count = df_latest['churn_prediction'].str.contains('Churn').sum() 
    churn_rate = churn_count / total_customers if total_customers > 0 else 0

    # Calcul des niveaux de risque (sur le dernier batch)
    risk_counts = df_latest['risk_level'].value_counts().reindex(['High', 'Medium', 'Low'], fill_value=0)

    # 3. AFFICHAGE DES CARTES (KPIs)
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Taux de Churn Prédit", f"{churn_rate*100:.1f}%", f"{churn_count} clients")
    col2.metric("Clients à Risque Élevé", risk_counts.get('High', 0))
    col3.metric("Clients à Risque Moyen", risk_counts.get('Medium', 0))
    col4.metric("Date du Dernier Batch", df_latest['prediction_date'].iloc[0]) # Date du premier client du dernier batch

    st.markdown("---")
    
    # 4. VISUALISATION DES DONNÉES
    
    st.subheader("Distribution des Niveaux de Risque")
    
    fig_risk = px.pie(
        names=risk_counts.index,
        values=risk_counts.values,
        title='Répartition des Niveaux de Risque',
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
        hole=.3
    )
    st.plotly_chart(fig_risk, use_container_width=True)
    

    st.markdown("---")
    
    # 5. TABLEAU DÉTAILLÉ DES PRÉDICTIONS
    st.subheader(f"Détail des Prédictions du Dernier Batch")
    
    # Rendre le DataFrame plus lisible pour l'affichage
    df_display_table = df_display_table[['customerID', 'churn_prediction', 'churn_probability', 'risk_level', 'prediction_date']].copy()
    df_display_table.columns = ['ID Client', 'Churn Prédit', 'Probabilité de Churn', 'Niveau de Risque', 'Date de Prédiction']
    
    # Filtrage par risque (pour le tableau)
    risk_filter = st.selectbox("Filtrer le Tableau par Niveau de Risque", ['Tous'] + list(df_display_table['Niveau de Risque'].unique()))

    if risk_filter != 'Tous':
        df_display_table = df_display_table[df_display_table['Niveau de Risque'] == risk_filter]

    st.dataframe(df_display_table, use_container_width=True, hide_index=True)