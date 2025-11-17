import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

# Configuration de la BDD
DB_NAME = "customers.db"
TABLE_NAME = "predictions_log"

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Dashboard Churn TeleConnect",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour charger les données
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
        st.error(f"Erreur de connexion à la BDD ou de lecture : {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

# Titre du Dashboard
# Titre du Dashboard
st.title("📊 Tableau de Bord de Prédiction du Churn")
st.markdown("Suivi en temps réel des prédictions de désabonnement client pour TeleConnect Afrique.")

df_predictions = load_data()

if df_predictions.empty:
    st.warning("Aucune donnée de prédiction trouvée dans la base de données. Lancez le Publisher pour générer des données.")
else:

    # 1. Identifier le timestamp du dernier batch
    latest_timestamp = df_predictions['batch_timestamp'].max()
    
    # 2. Filtrer les données pour ne garder que ce dernier batch
    df_latest = df_predictions[df_predictions['batch_timestamp'] == latest_timestamp].copy()
    
    # === REMPLACER df_predictions par df_latest dans tous les calculs suivants ===

    st.subheader(f"Statistiques Globales (Dernier Batch : {len(df_latest)} clients)")

    # 1. Calcul des Métriques Clés
    total_customers = len(df_latest)
    churn_count = df_latest['churn_prediction'].str.contains('Churn').sum() # Utiliser 'Churn' après correction
    churn_rate = churn_count / total_customers if total_customers > 0 else 0

    # Calcul des niveaux de risque
    risk_counts = df_predictions['risk_level'].value_counts().reindex(['High', 'Medium', 'Low'], fill_value=0)

    # Affichage des cartes (kpis)
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Taux de Churn Prédit", f"{churn_rate*100:.1f}%", f"{churn_count} clients")
    col2.metric("Clients à Risque Élevé", risk_counts.get('High', 0))
    col3.metric("Clients à Risque Moyen", risk_counts.get('Medium', 0))
    col4.metric("Dernière Prédiction", df_predictions['prediction_date'].iloc[0])

    st.markdown("---")
    
    # 2. Visualisation des Données
    
    st.subheader("Distribution des Niveaux de Risque")
    
    fig_risk = px.pie(
        names=risk_counts.index,
        values=risk_counts.values,
        title='Répartition des Niveaux de Risque',
        color_discrete_sequence=px.colors.sequential.RdBu # Rouge pour les risques élevés
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")
    
    # 3. Tableau Détaillé des Prédictions
    st.subheader("Détail des Prédictions du dernier batch")
    
    # Rendre le DataFrame plus lisible
    df_display = df_predictions[['customerID', 'churn_prediction', 'churn_probability', 'risk_level', 'prediction_date']].copy()
    df_display.columns = ['ID Client', 'Churn Prédit', 'Probabilité de Churn', 'Niveau de Risque', 'Date de Prédiction']
    
    # Filtrage par risque
    risk_filter = st.selectbox("Filtrer par Niveau de Risque", ['Tous'] + list(df_display['Niveau de Risque'].unique()))

    if risk_filter != 'Tous':
        df_display = df_display[df_display['Niveau de Risque'] == risk_filter]

    st.dataframe(df_display, use_container_width=True, hide_index=True)