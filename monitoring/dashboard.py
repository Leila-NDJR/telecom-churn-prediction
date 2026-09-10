"""
Dashboard TeleConnect Afrique — Prédiction du Churn Client
Auteur : K. Jessy

Multi-pages (navigation dans la barre latérale) :
  - Vue d'ensemble       : suivi des prédictions batch (base customers.db)
  - Simuler un client    : saisir un client et voir sa probabilité de churn,
                            calculée en appelant EXACTEMENT le même code que
                            l'API (api/main.py) — pas une logique dupliquée.
  - Modèles               : caractéristiques et comparaison des 4 modèles testés
  - Impact business       : simulation + méthodologie du calcul (pourquoi ça compte)
  - Recommandations       : implications professionnelles pour la direction
"""

import json
import os
import sqlite3
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Import de la logique de scoring RÉELLE de l'API (pas de duplication) :
# même feature engineering, même seuil business, mêmes recommandations que
# ce que renvoie /predict en production.
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from api.main import (  # noqa: E402
    BEST_THRESHOLD,
    THRESHOLD_SOURCE,
    CustomerData,
    calculate_clv_proxy,
    calculate_priority_score,
    feature_names,
    get_recommendations,
    metadata,
    model,
    post_process,
    preprocess_customer,
)

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
DB_NAME = os.path.join(BASE_DIR, "customers.db")
TABLE_NAME = "predictions_log"

# ==============================================================================
# 1. PALETTE PASTEL & THÈME
# ==============================================================================
PASTEL = {
    "violet": "#C4B5FD",       # violet clair — couleur d'accent principale
    "violet_deep": "#8B7CE0",  # violet un peu plus soutenu (texte sur fond clair)
    "violet_soft": "#F3F0FF",  # fond très clair
    "green": "#A8E6CF",        # succès / low risk
    "orange": "#FFD3A5",       # attention / medium risk
    "pink": "#FFAAA5",         # danger / high risk
    "blue": "#A7C7E7",         # secondaire
    "ink": "#000000",          # texte : noir
    "muted": "#8B87A0",
}
PLOTLY_PASTEL_SEQUENCE = [PASTEL["violet"], PASTEL["blue"], PASTEL["green"], PASTEL["orange"], PASTEL["pink"]]

st.set_page_config(
    page_title="Dashboard Churn TeleConnect",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #FDFCFF; }}
    /* Barre d'outils Streamlit par défaut : recolorée pour se fondre dans la
    page, SANS rien masquer (visibility/display) — un précédent essai avait
    caché par erreur le bouton pour rouvrir la barre latérale une fois
    réduite, qui vit dans ce même conteneur. */
    header[data-testid="stHeader"] {{ background-color: #FDFCFF; }}
    section[data-testid="stSidebar"] {{ background-color: {PASTEL["violet_soft"]}; }}
    /* Texte en noir, restreint aux éléments qui portent réellement du texte
    (pas de sélecteur `div` générique : ça avait aussi noirci des icônes -
    ex. le bouton de la barre latérale - rendues via `currentColor`, les
    rendant invisibles sur leur propre fond). Le logo JL reste blanc
    (règle .jl-logo plus bas, après celle-ci donc prioritaire à égalité de
    spécificité). */
    h1, h2, h3, h4, h5, h6, p, li, label, td, th,
    .stMarkdown, [data-testid="stCaptionContainer"],
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
        color: #000000 !important;
    }}
    div[data-testid="stMetric"] {{
        background-color: {PASTEL["violet_soft"]};
        border: 1px solid {PASTEL["violet"]};
        border-radius: 12px;
        padding: 12px 16px;
    }}
    .jl-logo, .jl-logo * {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 56px; height: 56px; border-radius: 14px;
        background: {PASTEL["violet"]};
        font-family: Georgia, 'Times New Roman', serif;
        font-style: italic; font-weight: 700; font-size: 24px;
        color: #FFFFFF !important; letter-spacing: 1px;
        box-shadow: 0 2px 8px rgba(139, 124, 224, 0.35);
    }}
    .jl-header {{ display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }}
    .jl-header-title {{ font-size: 15px; font-weight: 700; }}
    .jl-header-sub {{ font-size: 12px; }}
    .jl-card {{
        background: {PASTEL["violet_soft"]}; border-radius: 12px;
        padding: 16px 20px; border-left: 4px solid {PASTEL["violet"]};
        margin-bottom: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def jl_logo_header():
    st.sidebar.markdown(
        f"""
        <div class="jl-header">
            <div class="jl-logo">JL</div>
            <div>
                <div class="jl-header-title">K. Jessy</div>
                <div class="jl-header-sub">TeleConnect Afrique — Churn</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(html: str):
    st.markdown(f'<div class="jl-card">{html}</div>', unsafe_allow_html=True)


# ==============================================================================
# 2. CHARGEMENT DES DONNÉES PARTAGÉES
# ==============================================================================
@st.cache_data(ttl=60)
def load_predictions_log():
    """Charge l'historique des prédictions batch (base customers.db)."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY prediction_date DESC", conn)
    except Exception:
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()


@st.cache_data
def load_business_analysis():
    path = os.path.join(DATA_DIR, "business_analysis.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


business_analysis = load_business_analysis()

# ==============================================================================
# 3. NAVIGATION
# ==============================================================================
jl_logo_header()
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Vue d'ensemble",
        "🔍 Simuler un client",
        "🧠 Modèles",
        "💰 Impact business",
        "✅ Recommandations",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Modèle en production : **{metadata.get('best_model', 'N/A')}**")
st.sidebar.caption(f"Seuil de décision : **{BEST_THRESHOLD}** ({THRESHOLD_SOURCE})")

# ==============================================================================
# PAGE 1 — VUE D'ENSEMBLE
# ==============================================================================
if page == "📊 Vue d'ensemble":
    st.title("📊 Tableau de bord — Prédiction du churn")
    st.markdown("Suivi des prédictions de désabonnement client pour TeleConnect Afrique.")

    df_predictions = load_predictions_log()

    if df_predictions.empty:
        st.warning("Aucune donnée de prédiction trouvée dans `customers.db`. Lancez le publisher/subscriber MQTT pour générer des données, ou utilisez la page **Simuler un client** pour tester le modèle sans base de données.")
    else:
        latest_timestamp = df_predictions["prediction_date"].max()
        df_latest = df_predictions[df_predictions["prediction_date"] == latest_timestamp]

        st.caption(f"Historique total : {len(df_predictions)} prédictions enregistrées")
        st.subheader(f"Dernier batch ({len(df_latest)} clients)")

        total_customers = len(df_latest)
        churn_count = (df_latest["churn_prediction"] == "Churn").sum()
        churn_rate = churn_count / total_customers if total_customers > 0 else 0
        risk_counts = df_latest["risk_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Taux de churn prédit", f"{churn_rate*100:.1f}%", f"{churn_count} clients")
        col2.metric("Risque élevé", risk_counts.get("High", 0))
        col3.metric("Risque moyen", risk_counts.get("Medium", 0))
        col4.metric("Date du dernier batch", str(df_latest["prediction_date"].iloc[0])[:16])

        st.markdown("---")
        st.subheader("Répartition des niveaux de risque")
        fig_risk = px.pie(
            names=risk_counts.index,
            values=risk_counts.values,
            hole=0.55,
            color=risk_counts.index,
            color_discrete_map={"High": PASTEL["pink"], "Medium": PASTEL["orange"], "Low": PASTEL["green"]},
        )
        fig_risk.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=PASTEL["ink"]), legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("---")
        st.subheader("Détail des prédictions du dernier batch")
        df_display = df_latest[["customerID", "churn_prediction", "churn_probability", "risk_level", "prediction_date"]].copy()
        df_display.columns = ["ID Client", "Churn prédit", "Probabilité", "Niveau de risque", "Date"]
        risk_filter = st.selectbox("Filtrer par niveau de risque", ["Tous"] + list(df_display["Niveau de risque"].unique()))
        if risk_filter != "Tous":
            df_display = df_display[df_display["Niveau de risque"] == risk_filter]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

# ==============================================================================
# PAGE 2 — SIMULER UN CLIENT (scoring en direct, via le code réel de l'API)
# ==============================================================================
elif page == "🔍 Simuler un client":
    st.title("🔍 Simuler un client")
    st.markdown(
        "Entrez les caractéristiques d'un client pour voir **comment sa probabilité de churn est calculée**, "
        "en utilisant exactement le même code que l'API de production (`api/main.py`) — pas une approximation."
    )

    with st.expander("ℹ️ Comment ce score est-il calculé ?", expanded=False):
        st.markdown(
            f"""
1. **Feature engineering** — à partir des champs saisis, 5 variables sont dérivées : ancienneté catégorisée
   (`TenureCategory`), nombre total de services actifs, charge moyenne par service, facture moyenne par mois
   d'ancienneté, et un indicateur "senior avec famille".
2. **Encodage** — les variables catégorielles (contrat, méthode de paiement, services...) sont transformées
   en {len(feature_names)} colonnes numériques (one-hot encoding), alignées sur les colonnes vues à l'entraînement.
3. **Normalisation** — les valeurs sont mises à l'échelle avec le même `scaler` que celui utilisé à l'entraînement.
4. **Modèle** — le modèle **{metadata.get('best_model', 'N/A')}** calcule une probabilité de churn entre 0 et 1.
5. **Décision** — cette probabilité est comparée au seuil business **{BEST_THRESHOLD}** (pas 0.5 : ce seuil a été
   choisi car il maximise le revenu net attendu des actions de rétention, voir la page *Impact business*).
            """
        )

    with st.form("client_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Profil**")
            gender = st.selectbox("Genre", ["Female", "Male"])
            senior = st.selectbox("Senior (65+)", [0, 1], format_func=lambda x: "Oui" if x else "Non")
            partner = st.selectbox("A un(e) partenaire", ["No", "Yes"])
            dependents = st.selectbox("A des personnes à charge", ["No", "Yes"])
            tenure = st.number_input(
                "Ancienneté (mois)", min_value=0, max_value=600, value=12,
                help="Jusqu'à 600 mois (50 ans). Le jeu d'entraînement ne va pas au-delà de 72 mois, "
                     "mais toute ancienneté plus élevée reste correctement classée '4+ ans'.",
            )
        with c2:
            st.markdown("**Contrat & facturation**")
            contract = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Facturation sans papier", ["No", "Yes"])
            payment = st.selectbox(
                "Méthode de paiement",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly_charges = st.number_input("Facture mensuelle (FCFA)", min_value=0.0, value=70.0, step=5.0)
        with c3:
            st.markdown("**Services**")
            phone_service = st.selectbox("Service téléphonique", ["Yes", "No"])
            multiple_lines = st.selectbox("Lignes multiples", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Service internet", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Sécurité en ligne", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Support technique", ["No", "Yes", "No internet service"])

        c4, c5 = st.columns(2)
        with c4:
            online_backup = st.selectbox("Backup en ligne", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Protection appareil", ["No", "Yes", "No internet service"])
        with c5:
            streaming_tv = st.selectbox("TV en streaming", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Films en streaming", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("Calculer la probabilité de churn", type="primary")

    if submitted:
        try:
            customer = CustomerData(
                customerID="SIMULATION",
                gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dependents,
                tenure=tenure, Contract=contract, PaperlessBilling=paperless, PaymentMethod=payment,
                MonthlyCharges=monthly_charges, TotalCharges=monthly_charges * max(tenure, 1),
                PhoneService=phone_service, MultipleLines=multiple_lines, InternetService=internet_service,
                OnlineSecurity=online_security, OnlineBackup=online_backup, DeviceProtection=device_protection,
                TechSupport=tech_support, StreamingTV=streaming_tv, StreamingMovies=streaming_movies,
            )
            X = preprocess_customer(customer)
            probability = float(model.predict_proba(X)[0][1])
            prediction, risk_level, confidence = post_process(probability)
            customer_dict = customer.model_dump()
            clv_proxy = calculate_clv_proxy(customer_dict)
            priority_score = calculate_priority_score(probability, clv_proxy)
            recommendations = get_recommendations(probability, customer_dict)

            st.markdown("---")
            risk_color = {"High": PASTEL["pink"], "Medium": PASTEL["orange"], "Low": PASTEL["green"]}[risk_level]

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Probabilité de churn", f"{probability*100:.1f}%")
            r2.metric("Décision", prediction)
            r3.metric("Niveau de risque", risk_level)
            r4.metric("Valeur client estimée (CLV proxy)", f"{clv_proxy:,.0f} FCFA")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, BEST_THRESHOLD * 100], "color": PASTEL["violet_soft"]},
                        {"range": [BEST_THRESHOLD * 100, 75], "color": PASTEL["orange"]},
                        {"range": [75, 100], "color": PASTEL["pink"]},
                    ],
                    "threshold": {"line": {"color": PASTEL["ink"], "width": 3}, "value": BEST_THRESHOLD * 100},
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=PASTEL["ink"]))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.subheader("Recommandations pour ce client")
            for rec in recommendations:
                card(f"<b>{rec['rule']}</b> (priorité {rec['priority']}) — canal : {rec['channel']}<br>{rec['offer']}")

        except Exception as e:
            st.error(f"Erreur lors du calcul : {e}")

# ==============================================================================
# PAGE 3 — MODÈLES
# ==============================================================================
elif page == "🧠 Modèles":
    st.title("🧠 Caractéristiques des modèles")
    st.markdown("Comparaison des 4 modèles testés lors de l'entraînement, et détail du modèle retenu en production.")

    comparison = metadata.get("comparison", {})
    if comparison:
        comp_df = pd.DataFrame(comparison)
        comp_df = comp_df.set_index("Modèle") if "Modèle" in comp_df.columns else comp_df
        st.subheader("Comparaison des 4 modèles")
        st.dataframe(
            comp_df.style.format({c: "{:.3f}" for c in comp_df.columns if comp_df[c].dtype != bool}),
            use_container_width=True,
        )

        metric_choice = st.selectbox("Métrique à visualiser", ["AUC-ROC", "Recall", "Precision", "F1-Score", "Accuracy", "Overfitting"])
        fig_comp = px.bar(
            comp_df.reset_index(), x="Modèle", y=metric_choice, color="Modèle",
            color_discrete_sequence=PLOTLY_PASTEL_SEQUENCE, text_auto=".3f",
        )
        fig_comp.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=PASTEL["ink"]))
        st.plotly_chart(fig_comp, use_container_width=True)

        card(
            f"<b>Pourquoi {metadata.get('best_model', 'ce modèle')} ?</b> Il obtient le meilleur AUC-ROC "
            f"({comparison.get('AUC-ROC', {}).get('2', 0):.3f}) et le rappel le plus élevé "
            f"({comparison.get('Recall', {}).get('2', 0)*100:.1f}%) parmi les 4 modèles, avec un surapprentissage "
            f"contenu ({comparison.get('Overfitting', {}).get('2', 0)*100:.1f}%) — priorité donnée au rappel "
            "car manquer un client qui va churner coûte plus cher qu'une fausse alerte (voir page Impact business)."
        )

    st.markdown("---")
    st.subheader(f"Performance du modèle en production : {metadata.get('best_model', 'N/A')}")
    perf = metadata.get("performance", {})
    if perf:
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Accuracy", f"{perf.get('accuracy', 0)*100:.1f}%")
        p2.metric("Recall", f"{perf.get('recall', 0)*100:.1f}%")
        p3.metric("Precision", f"{perf.get('precision', 0)*100:.1f}%")
        p4.metric("F1-Score", f"{perf.get('f1_score', 0)*100:.1f}%")
        p5.metric("AUC-ROC", f"{perf.get('auc_roc', 0):.3f}")

    st.markdown("---")
    st.subheader("Variables les plus influentes")
    fi = metadata.get("feature_importance", {})
    if fi and "Feature" in fi:
        fi_df = pd.DataFrame({
            "Variable": list(fi["Feature"].values()),
            "Importance": list(fi["Importance"].values()),
        }).sort_values("Importance", ascending=True).tail(12)
        fig_fi = px.bar(
            fi_df, x="Importance", y="Variable", orientation="h",
            color_discrete_sequence=[PASTEL["violet"]],
        )
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=PASTEL["ink"]), height=450)
        st.plotly_chart(fig_fi, use_container_width=True)

# ==============================================================================
# PAGE 4 — IMPACT BUSINESS
# ==============================================================================
elif page == "💰 Impact business":
    st.title("💰 Impact business")

    st.markdown("### Pourquoi cette simulation est importante")
    card(
        "Un modèle avec un bon score technique (AUC, recall) n'a de valeur que s'il améliore un résultat "
        "financier réel. Deux erreurs coûtent différemment : <b>rater un client qui va churner (faux négatif)</b> "
        "fait perdre tout son revenu futur, alors qu'<b>alerter à tort sur un client fidèle (faux positif)</b> "
        "ne coûte que le prix d'une offre de rétention non nécessaire. Le seuil de décision "
        f"({BEST_THRESHOLD}, au lieu de 0.5 par défaut) a été choisi pour <b>maximiser le revenu net</b> "
        "compte tenu de cette asymétrie de coûts — c'est ce que cette page documente."
    )

    if business_analysis:
        cm = business_analysis.get("confusion_matrix", {})
        costs = business_analysis.get("costs", {})

        st.markdown("### Matrice de confusion au seuil optimal")
        cmc1, cmc2 = st.columns([1, 1.4])
        with cmc1:
            cm_df = pd.DataFrame(
                [[cm.get("tn", 0), cm.get("fp", 0)], [cm.get("fn", 0), cm.get("tp", 0)]],
                index=["Réel : reste", "Réel : churn"],
                columns=["Prédit : reste", "Prédit : churn"],
            )
            fig_cm = px.imshow(
                cm_df, text_auto=True, color_continuous_scale=[PASTEL["violet_soft"], PASTEL["violet"]],
            )
            fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=PASTEL["ink"]), height=350)
            st.plotly_chart(fig_cm, use_container_width=True)
        with cmc2:
            st.markdown(
                f"""
- **{cm.get('tp', 0)} vrais positifs** : clients à risque correctement identifiés → offre de rétention ciblée.
- **{cm.get('fn', 0)} faux négatifs** : clients qui churnent sans avoir été détectés → revenu perdu, coût = {costs.get('cost_fn_total', 0):,.0f} FCFA.
- **{cm.get('fp', 0)} faux positifs** : clients fidèles contactés inutilement → coût d'une offre non nécessaire, {costs.get('cost_fp_total', 0):,.0f} FCFA au total.
- **{cm.get('tn', 0)} vrais négatifs** : clients fidèles correctement ignorés, aucun coût.
                """
            )

        st.markdown("### Bilan financier au seuil optimal")
        b1, b2, b3 = st.columns(3)
        b1.metric("Revenu sauvé", f"{costs.get('revenue_saved', 0):,.0f} FCFA")
        b2.metric("Coût des actions", f"{costs.get('cost_actions', 0):,.0f} FCFA")
        b3.metric("Revenu net", f"{costs.get('net_revenue', 0):,.0f} FCFA", f"ROI {business_analysis.get('roi_percent', 0):.0f}%")

        st.markdown("### Pourquoi ce seuil précisément ?")
        ta = business_analysis.get("threshold_analysis", [])
        if ta:
            ta_df = pd.DataFrame(ta)
            fig_ta = px.line(
                ta_df, x="threshold", y="net_revenue", markers=True,
                labels={"threshold": "Seuil de décision", "net_revenue": "Revenu net (FCFA)"},
                color_discrete_sequence=[PASTEL["violet_deep"]],
            )
            fig_ta.add_vline(x=BEST_THRESHOLD, line_dash="dash", line_color=PASTEL["pink"],
                              annotation_text="Seuil retenu")
            fig_ta.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=PASTEL["ink"]))
            st.plotly_chart(fig_ta, use_container_width=True)
            st.caption(
                "Chaque point représente un seuil de décision testé (avant même de lancer une campagne réelle) "
                "et le revenu net qu'il aurait généré sur le jeu de test. Le seuil retenu est celui qui maximise "
                "cette courbe — ni trop bas (trop de fausses alertes), ni trop haut (trop de clients perdus)."
            )
    else:
        st.info("`business_analysis.json` introuvable — impossible d'afficher le détail des coûts réels.")

    st.markdown("---")
    st.subheader("🔮 Simulateur — vos propres hypothèses")
    st.caption("Ajustez les paramètres ci-dessous pour explorer d'autres scénarios que celui déjà optimisé ci-dessus.")

    df_predictions = load_predictions_log()
    if df_predictions.empty:
        st.info("Chargez des prédictions batch (page Vue d'ensemble) pour utiliser ce simulateur sur des données réelles, ou utilisez les chiffres de référence du seuil optimal ci-dessus.")
    else:
        latest_timestamp = df_predictions["prediction_date"].max()
        df_latest = df_predictions[df_predictions["prediction_date"] == latest_timestamp]
        total_customers = len(df_latest)
        churn_count = (df_latest["churn_prediction"] == "Churn").sum()
        churn_rate = churn_count / total_customers if total_customers > 0 else 0

        s1, s2 = st.columns(2)
        with s1:
            conversion_rate = st.slider("Taux de succès des campagnes (%)", 0, 100, 30,
                                          help="Part des clients à risque qui restent grâce à l'offre.") / 100
        with s2:
            retention_cost = st.number_input("Coût moyen d'une action de rétention (FCFA)", value=800, step=100)
            avg_revenue = 30600

        customers_saved = int(churn_count * conversion_rate)
        new_churn_count = churn_count - customers_saved
        new_churn_rate = new_churn_count / total_customers if total_customers > 0 else 0
        money_saved = customers_saved * avg_revenue
        campaign_cost = churn_count * retention_cost
        net_profit = money_saved - campaign_cost

        st.write(f"#### Résultat projeté : {new_churn_rate*100:.1f}% de churn (vs {churn_rate*100:.1f}% aujourd'hui)")
        res1, res2, res3 = st.columns(3)
        res1.metric("Clients sauvés", f"{customers_saved}")
        res2.metric("Chiffre d'affaires sauvé", f"{money_saved:,} FCFA")
        res3.metric("Résultat net", f"{net_profit:,} FCFA", f"coût campagne : {campaign_cost:,} FCFA")

        chart_data = pd.DataFrame({
            "Stade": ["Avant action", "Après action"],
            "Taux de churn (%)": [churn_rate * 100, new_churn_rate * 100],
        })
        fig_sim = px.bar(
            chart_data, x="Stade", y="Taux de churn (%)", color="Stade", text_auto=".1f",
            color_discrete_map={"Avant action": PASTEL["pink"], "Après action": PASTEL["green"]},
        )
        fig_sim.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=PASTEL["ink"]))
        st.plotly_chart(fig_sim, use_container_width=True)

# ==============================================================================
# PAGE 5 — RECOMMANDATIONS
# ==============================================================================
elif page == "✅ Recommandations":
    st.title("✅ Recommandations & implications professionnelles")
    st.markdown("Synthèse à destination d'une direction rétention / CRM, avant tout déploiement à grande échelle.")

    card(
        "<b>1. Valider la transférabilité sur de vraies données TeleConnect Afrique.</b><br>"
        "Le modèle est entraîné sur un jeu de données d'origine occidentale (structure de contrats/facturation "
        "type marché nord-américain). <i>Implication :</i> avant tout déploiement réel, ré-entraîner ou au "
        "minimum revalider les performances sur un échantillon de vrais clients TeleConnect — les leviers de "
        "churn (prépayé, multi-SIM, mobile money) peuvent différer significativement."
    )
    card(
        "<b>2. Mesurer l'efficacité réelle par un test A/B avant généralisation.</b><br>"
        "Le simulateur d'impact business (page précédente) utilise un taux de conversion des campagnes "
        "<i>hypothétique</i> (paramètre ajustable), pas mesuré sur le terrain. <i>Implication :</i> lancer une "
        "campagne pilote sur un échantillon contrôlé (groupe contacté vs groupe témoin) pour remplacer cette "
        "hypothèse par un chiffre réel avant d'engager un budget à grande échelle."
    )
    card(
        "<b>3. Réviser le seuil business trimestriellement.</b><br>"
        f"Le seuil actuel ({BEST_THRESHOLD}) est optimisé sur un jeu de données figé. <i>Implication :</i> les "
        "coûts (offre de rétention, revenu moyen par client) évoluent dans le temps — reproduire l'analyse "
        "coût/bénéfice (page Impact business) à intervalle régulier pour s'assurer que le seuil reste optimal."
    )
    card(
        "<b>4. Prioriser via le score risque × valeur, pas le risque seul.</b><br>"
        "Le `priority_score` (probabilité de churn × valeur client estimée) permet de concentrer un budget de "
        "rétention limité sur les clients où l'action a le plus d'impact. <i>Implication :</i> intégrer ce score "
        "dans l'outil CRM de l'équipe terrain plutôt que de traiter tous les clients à risque de la même façon."
    )
