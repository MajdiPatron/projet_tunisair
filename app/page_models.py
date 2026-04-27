"""
TUNISAIR — Streamlit App : Page Modèles & Évaluation
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "model")

TUNISAIR_RED  = "#E30613"
TUNISAIR_BLUE = "#1C3F6E"
COLORS = [TUNISAIR_RED, TUNISAIR_BLUE, "#00d97e"]


def render_models(styles):
    """Page évaluation & comparaison des modèles."""

    # Chargement résultats
    results_path = os.path.join(MODEL_DIR, "results.json")
    if not os.path.exists(results_path):
        st.warning("⚠️ Modèle non encore entraîné. Lancez d'abord `python run_training.py`")
        _render_instructions()
        return

    with open(results_path) as f:
        results = json.load(f)

    metrics     = results.get("metrics", [])
    cv_results  = results.get("cv_results", {})
    best_name   = results.get("best_model", "")
    feat_names  = results.get("feature_names", [])

    st.markdown(f"### 🏆 Meilleur Modèle : `{best_name}`")

    # ── TABLEAU COMPARAISON ───────────────────────────────────────────────
    st.markdown("<div class='section-card'><div class='section-title'>📊 Comparaison des Modèles</div>", unsafe_allow_html=True)

    df_metrics = pd.DataFrame(metrics).drop(columns=["confusion_matrix"], errors="ignore")
    df_metrics = df_metrics.set_index("model").round(4)

    # Coloriser la meilleure ligne
    def highlight_best(s):
        return ["background-color: rgba(227,6,19,0.2); font-weight:bold;"
                if s.name == best_name else "" for _ in s]

    st.dataframe(
        df_metrics.style.apply(highlight_best, axis=1)
                        .format("{:.4f}")
                        .background_gradient(cmap="RdYlGn", axis=0),
        width='stretch'
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── GRAPHIQUE COMPARAISON ─────────────────────────────────────────────
    st.markdown("<div class='section-card'><div class='section-title'>📈 Comparaison Visuelle des Métriques</div>", unsafe_allow_html=True)
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig = go.Figure()
    for i, m in enumerate(metrics):
        vals = [m.get(c, 0) for c in metric_cols]
        fig.add_trace(go.Bar(
            name=m["model"], x=[c.upper() for c in metric_cols], y=vals,
            marker_color=COLORS[i % len(COLORS)],
            text=[f"{v:.3f}" for v in vals],
            textposition="outside", textfont=dict(color="white", size=10),
            opacity=0.85
        ))
    layout = _dark_layout("Métriques par Modèle")
    layout["yaxis"] = dict(range=[0, 1.15], gridcolor="rgba(255,255,255,0.06)")
    layout["barmode"] = "group"
    fig.update_layout(**layout)
    st.plotly_chart(fig, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

    # ── MATRICES DE CONFUSION ─────────────────────────────────────────────
    st.markdown("<div class='section-card'><div class='section-title'>🔲 Matrices de Confusion</div>", unsafe_allow_html=True)
    cm_cols = st.columns(len(metrics))
    for col, m in zip(cm_cols, metrics):
        with col:
            cm = np.array(m["confusion_matrix"])
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=["Prédit: 0","Prédit: 1"], y=["Réel: 0","Réel: 1"],
                colorscale=[[0,"#0d1b2a"],[0.5,"#7a0010"],[1,TUNISAIR_RED]],
                text=cm, texttemplate="%{text}", showscale=False,
                textfont=dict(color="white", size=16)
            ))
            fig_cm.update_layout(
                title=dict(text=m["model"], font=dict(color="white", size=11)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), height=220,
                margin=dict(l=5, r=5, t=35, b=5),
                xaxis=dict(side="bottom"), yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_cm, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

    # ── CROSS-VALIDATION ─────────────────────────────────────────────────
    if cv_results:
        st.markdown("<div class='section-card'><div class='section-title'>🔄 Cross-Validation (5-Fold)</div>", unsafe_allow_html=True)
        cv_rows = []
        for model_name, cv_data in cv_results.items():
            cv_rows.append({
                "Modèle": model_name,
                "Accuracy (mean)":  f"{cv_data.get('accuracy',{}).get('mean',0):.4f}",
                "Accuracy (±std)":  f"±{cv_data.get('accuracy',{}).get('std',0):.4f}",
                "F1 (mean)":        f"{cv_data.get('f1',{}).get('mean',0):.4f}",
                "F1 (±std)":        f"±{cv_data.get('f1',{}).get('std',0):.4f}",
                "ROC-AUC (mean)":   f"{cv_data.get('roc_auc',{}).get('mean',0):.4f}",
                "ROC-AUC (±std)":   f"±{cv_data.get('roc_auc',{}).get('std',0):.4f}",
            })
        st.dataframe(pd.DataFrame(cv_rows), width='stretch', hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── FEATURE IMPORTANCE (CHARTS IMAGES) ───────────────────────────────
    st.markdown("<div class='section-card'><div class='section-title'>📌 Feature Importance</div>", unsafe_allow_html=True)
    fi_files = [f for f in os.listdir(REPORT_DIR) if f.startswith("feature_importance_")] if os.path.exists(REPORT_DIR) else []
    if fi_files:
        fi_cols = st.columns(min(3, len(fi_files)))
        for col, fname in zip(fi_cols, fi_files):
            with col:
                st.image(os.path.join(REPORT_DIR, fname), width='stretch')
    else:
        # Afficher à partir des résultats si features disponibles
        if feat_names and metrics:
            best_m = next((m for m in metrics if m["model"] == best_name), metrics[0])
            st.info("Lancez l'entraînement pour voir les graphiques de feature importance.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── SHAP ─────────────────────────────────────────────────────────────
    shap_path = os.path.join(REPORT_DIR, "shap_summary.png")
    if os.path.exists(shap_path):
        st.markdown("<div class='section-card'><div class='section-title'>🤖 Explicabilité IA — SHAP Values</div>", unsafe_allow_html=True)
        st.image(shap_path, width='stretch',
                 caption="Impact de chaque variable sur la prédiction du modèle")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── RAPPORT TEXTE ────────────────────────────────────────────────────
    with st.expander("📝 Rapport d'Évaluation Complet"):
        for m in metrics:
            st.markdown(f"#### [{m['model']}]")
            st.code(f"""
  Accuracy  : {m.get('accuracy',0):.4f}
  Precision : {m.get('precision',0):.4f}
  Recall    : {m.get('recall',0):.4f}
  F1-Score  : {m.get('f1',0):.4f}
  ROC-AUC   : {m.get('roc_auc',0):.4f}
            """)


def _render_instructions():
    st.markdown("""
    <div class='section-card'>
    <div class='section-title'>🚀 Instructions d'Entraînement</div>

    ```bash
    # 1. Installer les dépendances
    pip install -r requirements.txt

    # 2. Lancer le pipeline complet
    python run_training.py

    # 3. Démarrer l'application
    streamlit run app/streamlit_app.py
    ```
    </div>
    """, unsafe_allow_html=True)


def _dark_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(color="white", size=13)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.7)"),
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
