"""
TUNISAIR — Streamlit App : Page Prédiction
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.predict import predict_single, forecast_monthly
import plotly.graph_objects as go
import plotly.express as px

TUNISAIR_RED  = "#E30613"
TUNISAIR_BLUE = "#1C3F6E"


def render_prediction(model, scaler, feature_names, df_vis, styles, light_mode=False):
    """Page de prédiction interactive."""

    st.markdown("### 🔮 Prédiction de Rentabilité d'une Ligne")
    st.markdown("<p style='color:var(--text-muted);'>Renseignez les paramètres du vol pour obtenir une prédiction IA.</p>", unsafe_allow_html=True)

    # ── SIDEBAR FILTRES LIGNE ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**🛫 Filtres Ligne**")

        # Sélection ligne si colonne disponible
        ligne_col = next((c for c in df_vis.columns if "LIGNE" in c and df_vis[c].nunique() < 100), None)
        continent_col = next((c for c in df_vis.columns if "CONTINENT" in c or "REGION" in c), None)

        if continent_col:
            continents = ["Tous"] + sorted(df_vis[continent_col].dropna().unique().tolist())
            sel_continent = st.selectbox("Continent / Région", continents)
        else:
            sel_continent = "Tous"

        if ligne_col:
            if sel_continent != "Tous" and continent_col:
                lignes_dispo = df_vis[df_vis[continent_col] == sel_continent][ligne_col].dropna().unique().tolist()
            else:
                lignes_dispo = df_vis[ligne_col].dropna().unique().tolist()
            sel_ligne = st.selectbox("Ligne aérienne", ["-- Nouvelle --"] + sorted(lignes_dispo))
        else:
            sel_ligne = "-- Nouvelle --"

        # Pré-remplissage si ligne sélectionnée
        if sel_ligne != "-- Nouvelle --" and ligne_col:
            df_ligne = df_vis[df_vis[ligne_col] == sel_ligne]
            default_rev   = float(df_ligne["REVENUS"].mean()) if "REVENUS" in df_ligne else 500000
            default_cout  = float(df_ligne["COUTS"].mean())  if "COUTS" in df_ligne else 420000
            default_lf    = float(df_ligne["LOAD_FACTOR"].mean()) if "LOAD_FACTOR" in df_ligne else 0.75
            default_dist  = float(df_ligne["DISTANCE"].mean()) if "DISTANCE" in df_ligne else 1500.0
            default_cap   = float(df_ligne["CAPACITE"].mean()) if "CAPACITE" in df_ligne else 180.0
        else:
            default_rev, default_cout, default_lf = 500000.0, 420000.0, 0.75
            default_dist, default_cap = 1500.0, 180.0

    # ── FORMULAIRE PRÉDICTION ─────────────────────────────────────────────
    st.markdown("<div class='section-card'><div class='section-title'>⚙️ Paramètres du Vol</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        revenus = st.number_input("💵 Revenus estimés (TND)", min_value=0.0,
                                   value=default_rev, step=10000.0, format="%.0f")
        load_factor = st.slider("✈️ Load Factor (%)", min_value=0, max_value=100,
                                 value=int(default_lf * 100), step=1) / 100
    with col2:
        couts = st.number_input("💸 Coûts estimés (TND)", min_value=0.0,
                                 value=default_cout, step=10000.0, format="%.0f")
        distance = st.number_input("📏 Distance (km)", min_value=100.0,
                                    value=default_dist, step=100.0)
    with col3:
        capacite = st.number_input("🪑 Capacité (sièges)", min_value=50.0,
                                    value=default_cap, step=10.0)
        mois = st.selectbox("📅 Mois", list(range(1, 13)),
                             format_func=lambda m: {1:"Janvier",2:"Février",3:"Mars",4:"Avril",
                                                     5:"Mai",6:"Juin",7:"Juillet",8:"Août",
                                                     9:"Septembre",10:"Octobre",11:"Novembre",12:"Décembre"}[m])

    st.markdown("</div>", unsafe_allow_html=True)

    # ── MÉTRIQUES CALCULÉES EN DIRECT ─────────────────────────────────────
    profit_calc     = revenus - couts
    marge_calc      = (profit_calc / revenus * 100) if revenus > 0 else 0
    pax_est         = int(capacite * load_factor)
    rev_per_pax     = revenus / pax_est if pax_est > 0 else 0
    cost_per_km     = couts / distance if distance > 0 else 0

    st.markdown("**📊 Indicateurs calculés en temps réel**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Profit Estimé (TND)", f"{profit_calc:,.0f}", delta=f"{marge_calc:.1f}% marge")
    m2.metric("PAX Estimés", f"{pax_est:,}")
    m3.metric("Rev/PAX (TND)", f"{rev_per_pax:,.0f}")
    m4.metric("Coût/KM (TND)", f"{cost_per_km:.2f}")

    # ── BOUTON PRÉDIRE ────────────────────────────────────────────────────
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        predict_btn = st.button("🚀 LANCER LA PRÉDICTION", type="primary", width='stretch')

    if predict_btn:
        if model is None:
            st.error("❌ Modèle non chargé. Lancez d'abord run_training.py")
            return

        with st.spinner("🤖 Analyse IA en cours..."):
            input_data = {
                "REVENUS": revenus, "COUTS": couts,
                "LOAD_FACTOR": load_factor, "DISTANCE": distance,
                "CAPACITE": capacite, "MOIS": mois,
                "PAX": pax_est, "REV_PER_PAX": rev_per_pax,
                "COST_PER_KM": cost_per_km,
                "MARGE_OP": marge_calc / 100,
                "RATIO_COUT_REVENU": couts / revenus if revenus > 0 else 0,
                "HAUTE_SAISON": 1 if mois in [6, 7, 8, 12] else 0,
            }
            result = predict_single(input_data, model=model, scaler=scaler, feature_names=feature_names)

        # Résultat visuel
        st.markdown(styles.pred_result_box(
            result["label"], result["probabilite"],
            result["confiance"], result["profit_estime"]
        ), unsafe_allow_html=True)

        # Jauge probabilité
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result["probabilite"] * 100,
            title={"text": "Probabilité de Rentabilité (%)", "font": {"color": "var(--text-main)", "size": 14}},
            number={"suffix": "%", "font": {"color": "var(--text-main)", "size": 32}},
            delta={"reference": 50, "font": {"color": "var(--text-main)"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "var(--text-main)"},
                "bar":  {"color": "#00d97e" if result["prediction"] == 1 else "#ff4d6d"},
                "steps": [
                    {"range": [0, 40],  "color": "rgba(255,77,109,0.2)"},
                    {"range": [40, 60], "color": "rgba(255,255,255,0.05)"},
                    {"range": [60, 100],"color": "rgba(0,217,126,0.2)"},
                ],
                "threshold": {"line": {"color": "var(--text-main)", "width": 2}, "thickness": 0.8, "value": 50},
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="var(--text-main)"),
            height=280, margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_gauge, width='stretch')

        # Analyse what-if
        _render_whatif(revenus, couts, distance, load_factor, capacite, mois, model, scaler, feature_names, light_mode)

    # ── PRÉVISION MENSUELLE ────────────────────────────────────────────────
    st.markdown("---")
    _render_forecast(df_vis, model, scaler, feature_names, light_mode)


def _render_whatif(rev_base, cout_base, dist, lf, cap, mois, model, scaler, feature_names, light_mode):
    """Analyse scénarios what-if."""
    with st.expander("🔬 Analyse de Scénarios (What-If)"):
        st.markdown("**Comparez différents scénarios de revenus et coûts**")

        scenarios = {
            "Pessimiste (-20%)": {"REVENUS": rev_base * 0.8, "COUTS": cout_base * 1.1},
            "Base":              {"REVENUS": rev_base,        "COUTS": cout_base},
            "Optimiste (+20%)":  {"REVENUS": rev_base * 1.2,  "COUTS": cout_base * 0.95},
            "Best Case (+40%)":  {"REVENUS": rev_base * 1.4,  "COUTS": cout_base * 0.9},
        }

        rows = []
        for name, sc in scenarios.items():
            pax = int(cap * lf)
            inp = {
                **sc, "LOAD_FACTOR": lf, "DISTANCE": dist,
                "CAPACITE": cap, "MOIS": mois, "PAX": pax,
                "REV_PER_PAX": sc["REVENUS"]/pax if pax > 0 else 0,
                "COST_PER_KM": sc["COUTS"]/dist if dist > 0 else 0,
                "MARGE_OP": (sc["REVENUS"]-sc["COUTS"])/sc["REVENUS"] if sc["REVENUS"] > 0 else 0,
                "RATIO_COUT_REVENU": sc["COUTS"]/sc["REVENUS"] if sc["REVENUS"] > 0 else 0,
                "HAUTE_SAISON": 1 if mois in [6,7,8,12] else 0,
            }
            r = predict_single(inp, model=model, scaler=scaler, feature_names=feature_names)
            rows.append({
                "Scénario": name,
                "Revenus (TND)": f"{sc['REVENUS']:,.0f}",
                "Coûts (TND)":   f"{sc['COUTS']:,.0f}",
                "Profit (TND)":  f"{sc['REVENUS']-sc['COUTS']:,.0f}",
                "Probabilité":   f"{r['probabilite']*100:.1f}%",
                "Verdict":       r["label"],
            })

        df_sc = pd.DataFrame(rows)
        st.dataframe(df_sc, width='stretch', hide_index=True)

        # Graphique barres probabilités
        fig = go.Figure(go.Bar(
            x=[r["Scénario"] for r in rows],
            y=[float(r["Probabilité"].strip("%")) for r in rows],
            marker_color=["#ff4d6d" if float(r["Probabilité"].strip("%")) < 50 else "#00d97e" for r in rows],
            text=[r["Probabilité"] for r in rows],
            textposition="outside", textfont=dict(color="var(--text-main)"),
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="var(--text-main)", line_width=1)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="var(--text-main)"), height=300,
            title=dict(text="Probabilité de Rentabilité par Scénario", font=dict(color="var(--text-main)")),
            yaxis=dict(range=[0, 110], gridcolor="rgba(0,0,0,0.05)" if light_mode else "rgba(255,255,255,0.06)"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, width='stretch')


def _render_forecast(df_vis, model, scaler, feature_names, light_mode):
    """Prévision mensuelle."""
    st.markdown("### 📈 Prévision de Rentabilité — 6 Prochains Mois")

    if model is None:
        st.info("Modèle requis pour la prévision.")
        return

    n_months = st.slider("Horizon de prévision (mois)", 3, 12, 6)

    if st.button("🔮 Générer la Prévision"):
        with st.spinner("Calcul des prévisions..."):
            df_forecast = forecast_monthly(df_vis, n_months=n_months,
                                           model=model, scaler=scaler,
                                           feature_names=feature_names)

        if df_forecast.empty:
            st.warning("Données insuffisantes pour la prévision (nécessite colonnes MOIS et ANNEE).")
            return

        # Graphique
        colors = ["#00d97e" if l == "RENTABLE" else "#ff4d6d" for l in df_forecast["LABEL"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"M{int(r.MOIS):02d}/{int(r.ANNEE)}" for _, r in df_forecast.iterrows()],
            y=df_forecast["PROBABILITE"] * 100,
            marker_color=colors,
            name="Probabilité (%)",
            text=[f"{p*100:.0f}%" for p in df_forecast["PROBABILITE"]],
            textposition="outside", textfont=dict(color="var(--text-main)"),
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="var(--text-main)", line_width=1.5,
                      annotation_text="Seuil rentabilité", annotation_font_color="var(--text-main)")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="var(--text-main)"), height=350,
            title=dict(text=f"Prévision Rentabilité — {n_months} Mois", font=dict(color="var(--text-main)", size=14)),
            yaxis=dict(range=[0, 115], gridcolor="rgba(0,0,0,0.05)" if light_mode else "rgba(255,255,255,0.06)", title="Probabilité (%)"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, width='stretch')

        # Tableau
        df_forecast["Probabilité"] = df_forecast["PROBABILITE"].map(lambda x: f"{x*100:.1f}%")
        df_forecast["Profit Prédit"] = df_forecast["PROFIT_PRED"].map(lambda x: f"{x:,.0f} TND")
        st.dataframe(df_forecast[["MOIS","ANNEE","Profit Prédit","Probabilité","LABEL"]].rename(
            columns={"MOIS":"Mois","ANNEE":"Année","LABEL":"Verdict"}
        ), width='stretch', hide_index=True)
