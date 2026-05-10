"""
TUNISAIR — Streamlit App : Page Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TUNISAIR_RED  = "#E30613"
TUNISAIR_BLUE = "#1C3F6E"
DARK_BG       = "#0d1b2a"

def render_dashboard(df_vis: pd.DataFrame, styles, light_mode=False):
    """Affiche le dashboard principal avec KPIs et graphiques."""

    # ── KPIs ─────────────────────────────────────────────────────────────────
    profit_total = df_vis["PROFIT"].sum() if "PROFIT" in df_vis else 0
    rentable_pct = df_vis["RENTABLE"].mean() * 100 if "RENTABLE" in df_vis else 0
    lf_mean      = df_vis["LOAD_FACTOR"].mean() * 100 if "LOAD_FACTOR" in df_vis else 0
    rev_total    = df_vis["REVENUS"].sum() if "REVENUS" in df_vis else 0
    cout_total   = df_vis["COUTS"].sum() if "COUTS" in df_vis else 0
    n_vols       = len(df_vis)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(styles.kpi_card("💰", f"{profit_total/1e6:.1f}M", "Profit Total (TND)",
                                    delta=f"{rentable_pct:.0f}% rentables",
                                    positive=profit_total > 0), unsafe_allow_html=True)
    with col2:
        st.markdown(styles.kpi_card("📈", f"{rentable_pct:.1f}%", "Taux de Rentabilité",
                                    delta="Lignes bénéficiaires", positive=rentable_pct > 50),
                    unsafe_allow_html=True)
    with col3:
        st.markdown(styles.kpi_card("✈️", f"{lf_mean:.1f}%", "Load Factor Moyen",
                                    delta="Taux remplissage", positive=lf_mean > 70),
                    unsafe_allow_html=True)
    with col4:
        st.markdown(styles.kpi_card("🛫", f"{n_vols:,}", "Nombre de Vols",
                                    delta=f"Rev: {rev_total/1e6:.0f}M TND", positive=True),
                    unsafe_allow_html=True)

    st.markdown("<div class='red-divider'></div>", unsafe_allow_html=True)

    # ── GRAPHIQUES ROW 1 ─────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-card'><div class='section-title'>📊 Distribution des Profits</div>", unsafe_allow_html=True)
        if "PROFIT" in df_vis.columns:
            fig = px.histogram(df_vis, x="PROFIT", nbins=40, color_discrete_sequence=[TUNISAIR_RED])
            fig.add_vline(x=0, line_dash="dash", line_color="var(--text-main)", line_width=1.5)
            fig.update_layout(**_get_plot_layout("Distribution des Profits (TND)", light_mode))
            st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='section-card'><div class='section-title'>💹 Revenus vs Coûts</div>", unsafe_allow_html=True)
        if "REVENUS" in df_vis.columns and "COUTS" in df_vis.columns:
            sample = df_vis.sample(min(300, len(df_vis)), random_state=42)
            color_col = "RENTABLE" if "RENTABLE" in sample.columns else None
            fig = px.scatter(sample, x="COUTS", y="REVENUS",
                             color=color_col if color_col else None,
                             color_discrete_map={0: "#ff4d6d", 1: "#00d97e"},
                             opacity=0.6, size_max=8)
            # Droite y=x
            max_v = max(df_vis["REVENUS"].max(), df_vis["COUTS"].max())
            fig.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v],
                          mode="lines", line=dict(color="var(--text-main)", dash="dash", width=1.5),
                          name="Seuil rentabilité"))
            fig.update_layout(**_get_plot_layout("Revenus vs Coûts", light_mode))
            st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    # ── GRAPHIQUES ROW 2 ─────────────────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("<div class='section-card'><div class='section-title'>📅 Saisonnalité (Profit par Mois)</div>", unsafe_allow_html=True)
        if "MOIS" in df_vis.columns and "PROFIT" in df_vis.columns:
            monthly = df_vis.groupby("MOIS")["PROFIT"].mean().reset_index()
            monthly.columns = ["Mois", "Profit Moyen"]
            mois_names = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Jun",
                          7:"Jul",8:"Aoû",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}
            monthly["Mois"] = monthly["Mois"].map(mois_names)
            fig = px.bar(monthly, x="Mois", y="Profit Moyen",
                         color="Profit Moyen",
                         color_continuous_scale=[[0,"#ff4d6d"],[0.5,"#888"],[1,"#00d97e"]])
            fig.update_layout(**_get_plot_layout("Profit Moyen par Mois", light_mode))
            st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("<div class='section-card'><div class='section-title'>🎯 Répartition Rentabilité</div>", unsafe_allow_html=True)
        if "RENTABLE" in df_vis.columns:
            counts = df_vis["RENTABLE"].value_counts()
            labels = ["Non Rentable", "Rentable"]
            values = [counts.get(0, 0), counts.get(1, 0)]
            fig = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.6,
                marker_colors=[TUNISAIR_RED, "#00d97e"],
                textinfo="label+percent",
                textfont_size=13
            ))
            fig.add_annotation(text=f"{rentable_pct:.0f}%", x=0.5, y=0.5,
                                font_size=24, showarrow=False, font_color="var(--text-main)",
                                font=dict(weight="bold"))
            fig.update_layout(**_get_plot_layout("Répartition Rentabilité", light_mode))
            st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    # ── LOAD FACTOR ──────────────────────────────────────────────────────────
    if "LOAD_FACTOR" in df_vis.columns and "MOIS" in df_vis.columns:
        st.markdown("<div class='section-card'><div class='section-title'>✈️ Load Factor & Rentabilité par Mois</div>", unsafe_allow_html=True)
        monthly_lf = df_vis.groupby("MOIS").agg(
            Load_Factor=("LOAD_FACTOR","mean"),
            Rentabilite=("RENTABLE","mean")
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=monthly_lf["MOIS"], y=monthly_lf["Load_Factor"]*100,
                              name="Load Factor (%)", marker_color=TUNISAIR_BLUE, opacity=0.8))
        fig.add_trace(go.Scatter(x=monthly_lf["MOIS"], y=monthly_lf["Rentabilite"]*100,
                                  name="% Rentable", mode="lines+markers",
                                  line=dict(color=TUNISAIR_RED, width=2.5),
                                  marker=dict(size=8)), secondary_y=True)
        fig.update_layout(**_get_plot_layout("Load Factor et Rentabilité Mensuelle", light_mode))
        fig.update_yaxes(title_text="Load Factor (%)", secondary_y=False)
        fig.update_yaxes(title_text="% Lignes Rentables", secondary_y=True, color=TUNISAIR_RED)
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)


def render_data_explorer(df_vis: pd.DataFrame):
    """Explorateur de données interactif."""
    st.markdown("### 🔍 Explorateur de Données")

    col1, col2 = st.columns([1, 2])
    with col1:
        if "RENTABLE" in df_vis.columns:
            filtre = st.radio("Filtrer par", ["Tous","Rentables","Non Rentables"])
            if filtre == "Rentables":
                df_show = df_vis[df_vis["RENTABLE"] == 1]
            elif filtre == "Non Rentables":
                df_show = df_vis[df_vis["RENTABLE"] == 0]
            else:
                df_show = df_vis
        else:
            df_show = df_vis
        st.write(f"**{len(df_show):,}** enregistrements sélectionnés")

    with col2:
        cols_show = st.multiselect("Colonnes à afficher",
                                   options=df_vis.columns.tolist(),
                                   default=["REVENUS","COUTS","PROFIT","LOAD_FACTOR",
                                            "REV_PER_PAX","RENTABLE"][:min(6,len(df_vis.columns))])

    if cols_show:
        disp = df_show[cols_show].head(100)
        st.dataframe(disp.style.background_gradient(cmap="RdYlGn", subset=[c for c in cols_show if c in ["PROFIT","RENTABLE","LOAD_FACTOR"]]), width='stretch')

    # Stats descriptives
    with st.expander("📐 Statistiques Descriptives"):
        num_cols = df_vis.select_dtypes(include="number").columns.tolist()
        st.dataframe(df_vis[num_cols].describe().round(2), width='stretch')


def _get_plot_layout(title: str, light_mode: bool) -> dict:
    text_color = "#1a1a1a" if light_mode else "white"
    muted_color = "rgba(0,0,0,0.5)" if light_mode else "rgba(255,255,255,0.7)"
    grid_color = "rgba(0,0,0,0.05)" if light_mode else "rgba(255,255,255,0.06)"
    
    return dict(
        title=dict(text=title, font=dict(color=text_color, size=14, weight="bold")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=muted_color),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0.1)", bordercolor=grid_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, tickfont=dict(color=muted_color)),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, tickfont=dict(color=muted_color)),
    )
