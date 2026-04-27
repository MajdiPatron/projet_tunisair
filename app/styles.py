"""
TUNISAIR — Application Streamlit Premium
Styles CSS personnalisés (thème Tunisair)
"""

TUNISAIR_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* ── FOND GLOBAL ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #1a0a0f 100%);
    min-height: 100vh;
}

/* ── HEADER HERO ── */
.hero-header {
    background: linear-gradient(135deg, #E30613 0%, #c00010 40%, #1C3F6E 100%);
    padding: 2rem 3rem;
    border-radius: 0 0 24px 24px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex; align-items: center; gap: 2rem;
    box-shadow: 0 8px 32px rgba(227,6,19,0.4);
    position: relative; overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute; top: -50%; left: -10%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 70% 50%, rgba(255,255,255,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title { color: white; font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hero-sub   { color: rgba(255,255,255,0.75); font-size: 0.95rem; margin: 0.25rem 0 0; }
.hero-badge {
    background: rgba(255,255,255,0.15); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.3);
    padding: 0.4rem 1rem; border-radius: 20px;
    color: white; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px;
}

/* ── AVION ANIMATION ── */
.plane-anim {
    position: absolute; right: 3rem; top: 50%; transform: translateY(-50%);
    font-size: 4rem; animation: fly 3s ease-in-out infinite;
}
@keyframes fly {
    0%,100% { transform: translateY(-50%) rotate(-5deg) translateX(0); }
    50%      { transform: translateY(-60%) rotate(5deg) translateX(10px); }
}

/* ── KPI CARDS ── */
.kpi-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
    border-radius: 16px; padding: 1.5rem;
    transition: all 0.3s ease; cursor: default;
    text-align: center;
}
.kpi-card:hover {
    border-color: rgba(227,6,19,0.5);
    box-shadow: 0 8px 24px rgba(227,6,19,0.2);
    transform: translateY(-3px);
}
.kpi-icon   { font-size: 2rem; margin-bottom: 0.5rem; }
.kpi-value  { color: #ffffff; font-size: 1.8rem; font-weight: 800; }
.kpi-label  { color: rgba(255,255,255,0.55); font-size: 0.8rem; font-weight: 500;
               letter-spacing: 0.5px; text-transform: uppercase; margin-top: 0.25rem; }
.kpi-delta  { font-size: 0.85rem; margin-top: 0.4rem; font-weight: 600; }
.kpi-pos    { color: #00d97e; }
.kpi-neg    { color: #ff4d6d; }

/* ── PREDICTION RESULT ── */
.pred-box {
    border-radius: 20px; padding: 2.5rem;
    text-align: center; margin: 1.5rem 0;
    animation: fadeIn 0.6s ease;
}
.pred-box.rentable {
    background: linear-gradient(135deg, rgba(0,217,126,0.15) 0%, rgba(0,100,60,0.1) 100%);
    border: 2px solid rgba(0,217,126,0.4);
    box-shadow: 0 0 40px rgba(0,217,126,0.2);
}
.pred-box.non-rentable {
    background: linear-gradient(135deg, rgba(255,77,109,0.15) 0%, rgba(100,0,30,0.1) 100%);
    border: 2px solid rgba(255,77,109,0.4);
    box-shadow: 0 0 40px rgba(255,77,109,0.2);
}
.pred-icon  { font-size: 4rem; display: block; margin-bottom: 0.75rem; }
.pred-label { font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem; }
.pred-prob  { font-size: 1.1rem; color: rgba(255,255,255,0.7); }
.pred-conf  { display: inline-block; margin-top: 0.75rem; padding: 0.3rem 1rem;
               border-radius: 20px; font-size: 0.8rem; font-weight: 700; background: rgba(255,255,255,0.1); color: white; }

/* ── SECTION CARDS ── */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;
}
.section-title {
    color: #E30613; font-size: 1.1rem; font-weight: 700;
    letter-spacing: 0.5px; margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid rgba(227,6,19,0.3);
}

/* ── NAV SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a0e1a 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }

/* ── INPUTS ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider { filter: brightness(0.85); }

/* ── METRIC ── */
[data-testid="stMetricValue"] { color: white !important; font-weight: 800 !important; }

/* ── TABS ── */
.stTabs [role="tab"] { color: rgba(255,255,255,0.5) !important; font-weight: 600; }
.stTabs [role="tab"][aria-selected="true"] {
    color: #E30613 !important;
    border-bottom: 2px solid #E30613 !important;
}

/* ── PROGRESS BAR ── */
.prog-bar-outer { background: rgba(255,255,255,0.08); border-radius: 10px; height: 10px; margin: 0.3rem 0 1rem; }
.prog-bar-inner { height: 10px; border-radius: 10px; background: linear-gradient(90deg, #E30613, #ff6b7a); }

/* ── ANIMATIONS ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeIn 0.5s ease; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb { background: #E30613; border-radius: 3px; }

/* ── DIVIDER ── */
.red-divider {
    height: 2px;
    background: linear-gradient(90deg, #E30613, transparent);
    border: none; margin: 1.5rem 0;
}

/* ── TABLE ── */
.dataframe { background: transparent !important; }
.dataframe th { background: rgba(227,6,19,0.2) !important; color: white !important; }
.dataframe td { color: rgba(255,255,255,0.8) !important; border-color: rgba(255,255,255,0.05) !important; }
</style>
"""

def hero_header(logo_path=None):
    logo_html = ""
    if logo_path:
        import base64
        try:
            with open(logo_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{b64}" height="70" style="filter:brightness(0) invert(1);"/>'
        except Exception:
            logo_html = '<span style="font-size:3rem;">✈️</span>'
    else:
        logo_html = '<span style="font-size:3.5rem;">✈️</span>'

    return f"""
    <div class="hero-header">
        {logo_html}
        <div>
            <p class="hero-title">TUNISAIR Analytics</p>
            <p class="hero-sub">Système Intelligent de Prédiction de Rentabilité des Lignes</p>
            <span class="hero-badge">✈ AI-POWERED DECISION SYSTEM</span>
        </div>
        <div class="plane-anim">🛫</div>
    </div>"""

def kpi_card(icon, value, label, delta=None, positive=True):
    delta_html = ""
    if delta is not None:
        cls = "kpi-pos" if positive else "kpi-neg"
        arrow = "▲" if positive else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-card fade-in">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""

def pred_result_box(label, proba, confiance, profit):
    cls = "rentable" if label == "RENTABLE" else "non-rentable"
    icon = "✅" if label == "RENTABLE" else "❌"
    color = "#00d97e" if label == "RENTABLE" else "#ff4d6d"
    profit_fmt = f"{profit:,.0f} TND"
    return f"""
    <div class="pred-box {cls}">
        <span class="pred-icon">{icon}</span>
        <div class="pred-label" style="color:{color};">{label}</div>
        <div class="pred-prob">Probabilité de rentabilité : <b>{proba*100:.1f}%</b></div>
        <div class="pred-prob">Profit estimé : <b>{profit_fmt}</b></div>
        <span class="pred-conf">Confiance : {confiance}</span>
    </div>"""

def progress_bar(value, max_val=1.0, color="#E30613"):
    pct = min(100, value/max_val*100) if max_val != 0 else 0
    return f"""
    <div class="prog-bar-outer">
      <div class="prog-bar-inner" style="width:{pct:.1f}%;background:linear-gradient(90deg,{color},#ff6b7a);"></div>
    </div>"""
