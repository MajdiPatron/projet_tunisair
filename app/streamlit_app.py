"""
TUNISAIR — Application Streamlit PRINCIPALE
Point d'entrée : streamlit run app/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, sys, json, joblib, warnings

warnings.filterwarnings("ignore")

# ── Chemins ──────────────────────────────────────────────────────────────────
APP_DIR   = os.path.dirname(__file__)
ROOT_DIR  = os.path.join(APP_DIR, "..")
sys.path.insert(0, ROOT_DIR)

MODEL_DIR  = os.path.join(ROOT_DIR, "model")
DATA_DIR   = os.path.join(ROOT_DIR, "data")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")
LOGO_PATH  = os.path.join(ROOT_DIR, "Tunisair_(logo).png")

# ── Imports pages ─────────────────────────────────────────────────────────────
from app.styles import get_styles, hero_header, kpi_card, pred_result_box, progress_bar
import app.styles as styles
from app.page_dashboard  import render_dashboard
from app.page_prediction import render_prediction

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tunisair Analytics — Prédiction Rentabilité",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injecter CSS global
# Styles are injected after sidebar rendering to account for light_mode toggle


# ─────────────────────────────────────────────────────────────────────────────
# CACHE — CHARGEMENT DES DONNÉES ET MODÈLES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    """Charge le dataset de visualisation (pré-encodage)."""
    vis_path   = os.path.join(DATA_DIR, "dataset_vis.csv")
    final_path = os.path.join(DATA_DIR, "dataset_final.csv")

    if os.path.exists(vis_path):
        return pd.read_csv(vis_path)
    elif os.path.exists(final_path):
        return pd.read_csv(final_path)
    else:
        return _generate_demo_data()


@st.cache_resource(show_spinner=False)
def load_model_cached():
    """Charge le modèle et le scaler une seule fois."""
    model_path  = os.path.join(MODEL_DIR, "model_best.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    res_path    = os.path.join(MODEL_DIR, "results.json")

    model, scaler, feature_names = None, None, []

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # Correction pour l'incompatibilité de version scikit-learn (multi_class)
        if 'LogisticRegression' in str(type(model)) and not hasattr(model, 'multi_class'):
            setattr(model, 'multi_class', 'auto')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    if os.path.exists(res_path):
        with open(res_path) as f:
            results = json.load(f)
        feature_names = results.get("feature_names", [])

    return model, scaler, feature_names


def _generate_demo_data() -> pd.DataFrame:
    """Génère des données de démonstration si aucun CSV n'est trouvé."""
    np.random.seed(42)
    n = 500
    mois = np.random.randint(1, 13, n)
    haute_saison = np.isin(mois, [6, 7, 8, 12]).astype(int)
    dist = np.random.uniform(500, 5000, n)
    cap  = np.random.choice([100, 150, 180, 220, 280], n)
    lf   = np.clip(np.random.normal(0.72 + haute_saison * 0.08, 0.12, n), 0.3, 1.0)
    pax  = (cap * lf).astype(int)
    rev  = pax * np.random.uniform(250, 600, n) + haute_saison * 50000
    cout = dist * np.random.uniform(80, 150, n) + cap * 500
    profit = rev - cout

    df = pd.DataFrame({
        "MOIS": mois, "ANNEE": np.random.choice([2022,2023,2024], n),
        "HAUTE_SAISON": haute_saison,
        "DISTANCE": dist, "CAPACITE": cap,
        "LOAD_FACTOR": lf, "PAX": pax,
        "REVENUS": rev, "COUTS": cout,
        "PROFIT": profit,
        "REV_PER_PAX": rev / np.maximum(pax, 1),
        "COST_PER_KM": cout / np.maximum(dist, 1),
        "MARGE_OP": profit / np.maximum(rev, 1),
        "RATIO_COUT_REVENU": cout / np.maximum(rev, 1),
        "RENTABLE": (profit > 0).astype(int),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=140)
        else:
            st.markdown("# ✈️ TUNISAIR")

        st.markdown("---")
        
        # Thème Toggle
        light_mode = st.toggle("☀️ Mode Clair", value=False)
        st.markdown("---")

        st.markdown("### 🧭 Navigation")

        page = st.radio(
            label="Menu",
            options=["🏠 Dashboard", "🔮 Prédiction"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Statut système
        model, scaler, feature_names = load_model_cached()
        model_status = "✅ Chargé" if model else "⚠️ Non entraîné"
        data_exists  = os.path.exists(os.path.join(DATA_DIR, "dataset_final.csv"))
        data_status  = "✅ Disponibles" if data_exists else "⚠️ Données demo"

        st.markdown("### ⚙️ Statut Système")
        st.markdown(f"""
        <div class="status-box">
        🤖 Modèle : <b>{model_status}</b><br/>
        📂 Données : <b>{data_status}</b><br/>
        🐍 Streamlit : <b>✅ Actif</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Bouton entraînement
        if not model:
            st.markdown("""
            <div style="background:rgba(227,6,19,0.1);border:1px solid rgba(227,6,19,0.3);
                        border-radius:10px;padding:0.8rem;font-size:0.8rem;color:rgba(255,255,255,0.8);">
            ⚡ Pour activer la prédiction IA :<br/><br/>
            <code style="color:#E30613;">python run_training.py</code>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "<p style='color:var(--text-muted);font-size:0.7rem;text-align:center;'>"
            "© 2024 Tunisair Analytics<br/>Powered by XGBoost & SHAP</p>",
            unsafe_allow_html=True
        )

    return page, light_mode


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Navigation & Theme
    page, light_mode = render_sidebar()

    # Injecter CSS global
    st.markdown(get_styles(light_mode), unsafe_allow_html=True)

    # Header hero
    st.markdown(hero_header(LOGO_PATH), unsafe_allow_html=True)

    # Chargement données & modèle
    with st.spinner("⏳ Chargement des données..."):
        df_vis = load_data()

    model, scaler, feature_names = load_model_cached()

    # ── ROUTING ──────────────────────────────────────────────────────────────
    if page == "🏠 Dashboard":
        render_dashboard(df_vis, styles, light_mode)

    elif page == "🔮 Prédiction":
        render_prediction(model, scaler, feature_names, df_vis, styles, light_mode)


if __name__ == "__main__":
    main()
