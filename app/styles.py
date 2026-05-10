"""
TUNISAIR — Application Streamlit Premium
Styles CSS personnalisés (thème Tunisair) avec support Mode Clair/Sombre
"""

def get_styles(light_mode=False):
    # Variables de design
    if light_mode:
        bg_gradient = "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)"
        sidebar_bg  = "linear-gradient(180deg, #ffffff 0%, #f0f2f6 100%)"
        surface     = "rgba(255, 255, 255, 0.7)"
        border      = "rgba(227, 6, 19, 0.15)"
        text_main   = "#1a1a1a"
        text_muted  = "rgba(0, 0, 0, 0.6)"
        card_shadow = "0 10px 30px rgba(0,0,0,0.05)"
        input_bg    = "#ffffff"
    else:
        bg_gradient = "linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #1a0a0f 100%)"
        sidebar_bg  = "linear-gradient(180deg, #0d1b2a 0%, #0a0e1a 100%)"
        surface     = "rgba(255, 255, 255, 0.04)"
        border      = "rgba(255, 255, 255, 0.08)"
        text_main   = "#ffffff"
        text_muted  = "rgba(255, 255, 255, 0.6)"
        card_shadow = "0 10px 40px rgba(0,0,0,0.3)"
        input_bg    = "rgba(255, 255, 255, 0.03)"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {{
        --tunisair-red: #E30613;
        --tunisair-blue: #1C3F6E;
        --surface: {surface};
        --border: {border};
        --text-main: {text_main};
        --text-muted: {text_muted};
        --card-shadow: {card_shadow};
    }}

    * {{ font-family: 'Plus Jakarta Sans', sans-serif; }}

    /* ── FOND GLOBAL ── */
    .stApp {{
        background: {bg_gradient};
        color: var(--text-main);
    }}

    /* ── HEADER HERO ── */
    .hero-header {{
        background: linear-gradient(135deg, var(--tunisair-red) 0%, #c00010 40%, var(--tunisair-blue) 100%);
        padding: 2.5rem 3rem;
        border-radius: 0 0 32px 32px;
        margin: -1rem -1rem 2.5rem -1rem;
        display: flex; align-items: center; gap: 2.5rem;
        box-shadow: 0 15px 45px rgba(227,6,19,0.35);
        position: relative; overflow: hidden;
    }}
    .hero-header::before {{
        content: '';
        position: absolute; top: -50%; left: -10%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 70% 50%, rgba(255,255,255,0.1) 0%, transparent 60%);
        pointer-events: none;
    }}
    .hero-title {{ color: white; font-size: 2.5rem; font-weight: 800; margin: 0; letter-spacing: -1px; }}
    .hero-sub   {{ color: rgba(255,255,255,0.85); font-size: 1rem; font-weight: 500; margin: 0.5rem 0 0; }}
    .hero-badge {{
        background: rgba(255,255,255,0.2); backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 0.5rem 1.2rem; border-radius: 30px;
        color: white; font-size: 0.8rem; font-weight: 700; letter-spacing: 1.5px;
        display: inline-block; margin-top: 1rem;
    }}

    /* ── AVION ANIMATION ── */
    .plane-anim {{
        position: absolute; right: 4rem; top: 50%; transform: translateY(-50%);
        font-size: 5rem; animation: fly 4s ease-in-out infinite;
    }}
    @keyframes fly {{
        0%,100% {{ transform: translateY(-50%) rotate(-5deg) translateX(0); }}
        50%      {{ transform: translateY(-70%) rotate(5deg) translateX(20px); }}
    }}

    /* ── KPI CARDS (GLASSMORPHISM) ── */
    .kpi-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        backdrop-filter: blur(15px);
        border-radius: 24px; padding: 1.8rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: var(--card-shadow);
        text-align: center;
    }}
    .kpi-card:hover {{
        border-color: var(--tunisair-red);
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(227,6,19,0.15);
    }}
    .kpi-icon   {{ font-size: 2.5rem; margin-bottom: 0.8rem; }}
    .kpi-value  {{ color: var(--text-main); font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; }}
    .kpi-label  {{ color: var(--text-muted); font-size: 0.8rem; font-weight: 700;
                   letter-spacing: 1px; text-transform: uppercase; margin-top: 0.5rem; }}
    .kpi-delta  {{ font-size: 0.9rem; margin-top: 0.6rem; font-weight: 700; padding: 0.2rem 0.8rem; border-radius: 12px; display: inline-block; }}
    .kpi-pos    {{ color: #00d97e; background: rgba(0,217,126,0.1); }}
    .kpi-neg    {{ color: #ff4d6d; background: rgba(255,77,109,0.1); }}

    /* ── PREDICTION RESULT ── */
    .pred-box {{
        border-radius: 28px; padding: 3rem;
        text-align: center; margin: 2rem 0;
        animation: slideInUp 0.7s ease-out;
        backdrop-filter: blur(20px);
    }}
    .pred-box.rentable {{
        background: linear-gradient(135deg, rgba(0,217,126,0.12) 0%, rgba(0,100,60,0.05) 100%);
        border: 2px solid rgba(0,217,126,0.3);
        box-shadow: 0 0 50px rgba(0,217,126,0.15);
    }}
    .pred-box.non-rentable {{
        background: linear-gradient(135deg, rgba(255,77,109,0.12) 0%, rgba(100,0,30,0.05) 100%);
        border: 2px solid rgba(255,77,109,0.3);
        box-shadow: 0 0 50px rgba(255,77,109,0.15);
    }}
    .pred-icon  {{ font-size: 5rem; display: block; margin-bottom: 1rem; filter: drop-shadow(0 5px 15px rgba(0,0,0,0.1)); }}
    .pred-label {{ font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem; letter-spacing: -1px; }}
    .pred-prob  {{ font-size: 1.2rem; color: var(--text-main); font-weight: 500; opacity: 0.9; margin: 0.5rem 0; }}
    .pred-conf  {{ display: inline-block; margin-top: 1.2rem; padding: 0.5rem 1.5rem;
                   border-radius: 25px; font-size: 0.85rem; font-weight: 800; 
                   background: var(--tunisair-blue); color: white; box-shadow: 0 5px 15px rgba(28,63,110,0.3); }}

    /* ── SECTION CARDS ── */
    .section-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 24px; padding: 2rem; margin-bottom: 2rem;
        box-shadow: var(--card-shadow);
    }}
    .section-title {{
        color: var(--tunisair-red); font-size: 1.3rem; font-weight: 800;
        letter-spacing: -0.5px; margin-bottom: 1.5rem;
        display: flex; align-items: center; gap: 0.75rem;
    }}
    .section-title::after {{
        content: ''; flex: 1; height: 1px; 
        background: linear-gradient(90deg, rgba(227,6,19,0.3), transparent);
    }}

    /* ── NAV SIDEBAR ── */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
        border-right: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] * {{ color: var(--text-main) !important; }}
    
    /* ── SIDEBAR STATUT BOX ── */
    .status-box {{
        background: rgba(227,6,19,0.05);
        border: 1px solid rgba(227,6,19,0.1);
        border-radius: 16px; padding: 1.2rem;
        font-size: 0.85rem; line-height: 1.6;
    }}

    /* ── INPUTS ── */
    .stSelectbox > div > div, .stNumberInput > div > div > input {{
        background-color: {input_bg} !important;
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        color: var(--text-main) !important;
    }}

    /* ── METRIC ── */
    [data-testid="stMetricValue"] {{ color: var(--text-main) !important; font-weight: 800 !important; font-size: 2.2rem !important; }}
    [data-testid="stMetricLabel"] {{ color: var(--text-muted) !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 1px; }}

    /* ── TABS ── */
    .stTabs [role="tab"] {{ color: var(--text-muted) !important; font-weight: 700; font-size: 1rem; border: none !important; }}
    .stTabs [role="tab"][aria-selected="true"] {{
        color: var(--tunisair-red) !important;
        background: rgba(227,6,19,0.05) !important;
        border-radius: 12px 12px 0 0 !important;
    }}

    /* ── PROGRESS BAR ── */
    .prog-bar-outer {{ background: rgba(0,0,0,0.05); border-radius: 12px; height: 14px; margin: 0.5rem 0 1.5rem; overflow: hidden; }}
    .prog-bar-inner {{ height: 14px; border-radius: 12px; background: linear-gradient(90deg, var(--tunisair-red), #ff6b7a); box-shadow: 0 0 10px rgba(227,6,19,0.3); }}

    /* ── ANIMATIONS ── */
    @keyframes slideInUp {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .fade-in {{ animation: slideInUp 0.6s ease-out; }}

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: var(--tunisair-red); border-radius: 10px; border: 2px solid transparent; background-clip: content-box; }}

    /* ── TABLE ── */
    .dataframe {{ border: none !important; border-collapse: separate !important; border-spacing: 0 8px !important; }}
    .dataframe th {{ background: var(--tunisair-blue) !important; color: white !important; border: none !important; padding: 12px !important; border-radius: 8px; }}
    .dataframe td {{ background: var(--surface) !important; color: var(--text-main) !important; border: none !important; padding: 12px !important; }}
    .dataframe tr:hover td {{ background: rgba(227,6,19,0.05) !important; }}
    </style>
    """

def hero_header(logo_path=None):
    logo_html = ""
    if logo_path:
        import base64
        try:
            with open(logo_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{b64}" height="85" style="filter:brightness(0) invert(1);"/>'
        except Exception:
            logo_html = '<span style="font-size:3.5rem;">✈️</span>'
    else:
        logo_html = '<span style="font-size:4rem;">✈️</span>'

    return f"""
    <div class="hero-header">
        {logo_html}
        <div>
            <p class="hero-title">TUNISAIR Analytics</p>
            <p class="hero-sub">Plateforme Prédictive de Rentabilité des Lignes Aériennes</p>
            <span class="hero-badge">✨ PREMIUM AI INTELLIGENCE</span>
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
    icon = "🛡️" if label == "RENTABLE" else "⚠️"
    color = "#00d97e" if label == "RENTABLE" else "#ff4d6d"
    profit_fmt = f"{profit:,.0f} TND"
    return f"""
    <div class="pred-box {cls}">
        <span class="pred-icon">{icon}</span>
        <div class="pred-label" style="color:{color};">{label}</div>
        <div class="pred-prob">Indice de rentabilité : <b>{proba*100:.1f}%</b></div>
        <div class="pred-prob">Impact financier estimé : <b style="color:{color};">{profit_fmt}</b></div>
        <span class="pred-conf">Niveau de Confiance : {confiance}</span>
    </div>"""

def progress_bar(value, max_val=1.0, color="#E30613"):
    pct = min(100, value/max_val*100) if max_val != 0 else 0
    return f"""
    <div class="prog-bar-outer">
      <div class="prog-bar-inner" style="width:{pct:.1f}%;background:linear-gradient(90deg,{color},#ff6b7a);"></div>
    </div>"""
