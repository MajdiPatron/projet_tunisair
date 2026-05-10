"""
TUNISAIR - Module de Prédiction & Explicabilité (SHAP)
CRISP-DM Phase 6: Deployment
"""
import pandas as pd
import numpy as np
import os, joblib, json, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "model")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

# ─── CHARGEMENT DU MODÈLE ────────────────────────────────────────────────────
def load_model(model_name: str = "model_best.pkl"):
    path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    model = joblib.load(path)
    
    # Correction pour l'incompatibilité de version scikit-learn (multi_class)
    if 'LogisticRegression' in str(type(model)) and not hasattr(model, 'multi_class'):
        setattr(model, 'multi_class', 'auto')
        
    return model

def load_scaler():
    path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

def load_results() -> dict:
    path = os.path.join(MODEL_DIR, "results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def load_feature_names() -> list:
    results = load_results()
    return results.get("feature_names", [])


# ─── PRÉDICTION UNITAIRE ─────────────────────────────────────────────────────
def predict_single(input_dict: dict, model=None, scaler=None,
                   feature_names: list = None) -> dict:
    """
    Prédit la rentabilité pour un vol donné.

    Args:
        input_dict: {"REVENUS": 500000, "COUTS": 420000, ...}
        model: modèle sklearn/xgboost chargé
        scaler: StandardScaler
        feature_names: liste des features du modèle

    Returns:
        {"prediction": 1, "probabilite": 0.82, "label": "RENTABLE", "confiance": "Haute"}
    """
    if model is None:      model = load_model()
    if scaler is None:     scaler = load_scaler()
    if feature_names is None: feature_names = load_feature_names()

    # Construire le vecteur de features
    row = {f: input_dict.get(f, 0) for f in feature_names}

    # Calcul des ratios dérivés si non fournis
    rev  = input_dict.get("REVENUS", 0)
    cout = input_dict.get("COUTS", 0)
    pax  = input_dict.get("PAX", input_dict.get("NB_PAX", 100))
    dist = input_dict.get("DISTANCE", 1000)
    
    # Valeurs par défaut pour éviter les outliers du scaler (ex: ANNEE=0)
    import datetime
    now = datetime.datetime.now()
    default_year  = input_dict.get("ANNEE", now.year)
    default_month = input_dict.get("MOIS", input_dict.get("MOIS_NUM", now.month))

    if "PROFIT" in feature_names:
        row["PROFIT"] = rev - cout
    if "MARGE_OP" in feature_names:
        row["MARGE_OP"] = (rev - cout) / rev if rev > 0 else 0
    if "RATIO_COUT_REVENU" in feature_names:
        row["RATIO_COUT_REVENU"] = cout / rev if rev > 0 else 0
    if "REV_PER_PAX" in feature_names:
        row["REV_PER_PAX"] = rev / pax if pax > 0 else 0
    if "COST_PER_KM" in feature_names:
        row["COST_PER_KM"] = cout / dist if dist > 0 else 0
    if "ANNEE" in feature_names:
        row["ANNEE"] = default_year
    if "MOIS" in feature_names:
        row["MOIS"] = default_month
    if "MOIS_NUM" in feature_names:
        row["MOIS_NUM"] = default_month

    # Debug: afficher le vecteur final pour diagnostic
    # print(f"DEBUG Prediction Input: {row}")

    X = pd.DataFrame([row])[feature_names].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # Appliquer scaler pour LR, sinon brut
    try:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names) if scaler else X
    except Exception:
        X_scaled = X

    # Prédiction
    try:
        proba = model.predict_proba(X_scaled)[0][1]
    except Exception:
        proba = model.predict_proba(X)[0][1]

    # --- LOGIQUE MÉTIER (Sanity Check) ---
    # Si le profit calculé est négatif, la ligne ne peut pas être "Rentable" 
    # quelle que soit la prédiction statistique du modèle (qui peut être biaisé).
    profit_calc = rev - cout
    if profit_calc <= 0:
        pred = 0
        label = "NON RENTABLE"
        # On ajuste la probabilité pour refléter le risque
        proba = min(proba, 0.49) 
    else:
        pred = int(proba >= 0.5)
        label = "RENTABLE" if pred == 1 else "NON RENTABLE"

    confiance = "Très Haute" if proba > 0.85 or proba < 0.15 else \
                "Haute" if proba > 0.70 or proba < 0.30 else "Modérée"

    return {
        "prediction":  pred,
        "probabilite": float(proba),
        "label":       label,
        "confiance":   confiance,
        "profit_estime": rev - cout,
    }


# ─── PRÉDICTION EN BATCH ─────────────────────────────────────────────────────
def predict_batch(df: pd.DataFrame, model=None, scaler=None,
                  feature_names: list = None) -> pd.DataFrame:
    """Prédiction sur un DataFrame complet."""
    if model is None:         model = load_model()
    if scaler is None:        scaler = load_scaler()
    if feature_names is None: feature_names = load_feature_names()

    available = [f for f in feature_names if f in df.columns]
    X = df[available].fillna(0).replace([np.inf,-np.inf], 0)

    # Compléter les colonnes manquantes
    for f in feature_names:
        if f not in X.columns:
            X[f] = 0
    X = X[feature_names]

    try:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names) if scaler else X
    except Exception:
        X_scaled = X

    try:
        probas = model.predict_proba(X_scaled)[:, 1]
    except Exception:
        probas = model.predict_proba(X)[:, 1]

    df_result = df.copy()
    df_result["PROBABILITE_RENTABILITE"] = probas
    df_result["PREDICTION"] = (probas >= 0.5).astype(int)
    df_result["LABEL"] = df_result["PREDICTION"].map({1:"RENTABLE", 0:"NON RENTABLE"})
    return df_result


# ─── SHAP EXPLICABILITÉ ──────────────────────────────────────────────────────
def compute_shap_values(model, X: pd.DataFrame, feature_names: list = None,
                        max_samples: int = 200):
    """
    Calcule les SHAP values pour l'explicabilité du modèle.

    Returns:
        shap_values array ou None si SHAP non disponible
    """
    try:
        import shap
        X_sample = X.head(max_samples).fillna(0).replace([np.inf,-np.inf], 0)

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_sample)

        shap_values = explainer.shap_values(X_sample)

        # Pour classif binaire, prendre classe 1
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        return shap_values, X_sample, explainer
    except Exception as e:
        print(f"  ⚠️  SHAP non disponible : {e}")
        return None, None, None


def plot_shap_summary(model, X: pd.DataFrame, feature_names: list,
                      save_path: str = None) -> str:
    """Génère et sauvegarde le plot SHAP summary."""
    try:
        import shap
        shap_vals, X_s, _ = compute_shap_values(model, X, feature_names)
        if shap_vals is None:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_s, feature_names=feature_names,
                          show=False, plot_size=None)
        plt.title("SHAP — Impact des Variables sur la Prédiction",
                  fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(REPORT_DIR, "shap_summary.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path
    except Exception as e:
        print(f"  ⚠️  SHAP plot échoué : {e}")
        return None


def plot_shap_waterfall(model, X_row: pd.DataFrame, feature_names: list,
                        save_path: str = None) -> str:
    """Waterfall pour une prédiction individuelle."""
    try:
        import shap
        shap_vals, X_s, explainer = compute_shap_values(model, X_row, feature_names, max_samples=1)
        if shap_vals is None:
            return None

        fig = plt.figure(figsize=(10, 5))
        shap.waterfall_plot(shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value if hasattr(explainer,"expected_value") else 0,
            data=X_s.iloc[0].values,
            feature_names=feature_names
        ), show=False)
        plt.title("SHAP Waterfall — Explication Prédiction", fontweight="bold")
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(REPORT_DIR, "shap_waterfall.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path
    except Exception as e:
        print(f"  ⚠️  Waterfall échoué : {e}")
        return None


# ─── PRÉVISION MENSUELLE (FORECAST) ──────────────────────────────────────────
def forecast_monthly(df_hist: pd.DataFrame, n_months: int = 6,
                     model=None, scaler=None, feature_names=None) -> pd.DataFrame:
    """
    Prédit la rentabilité pour les n_months prochains mois
    en extrapolant tendances des revenus et coûts.
    """
    if "MOIS" not in df_hist.columns or "ANNEE" not in df_hist.columns:
        return pd.DataFrame()

    agg_dict = {"REVENUS": "sum", "COUTS": "sum"}
    if "PAX" in df_hist.columns:
        agg_dict["PAX"] = "sum"
    if "LOAD_FACTOR" in df_hist.columns:
        agg_dict["LOAD_FACTOR"] = "mean"
    monthly = df_hist.groupby(["ANNEE", "MOIS"]).agg(agg_dict).reset_index()

    if monthly.empty or len(monthly) < 3:
        return pd.DataFrame()

    # Tendance linéaire simple
    n = len(monthly)
    rev_trend  = np.polyfit(range(n), monthly["REVENUS"].values, 1)
    cout_trend = np.polyfit(range(n), monthly["COUTS"].values, 1)

    rows = []
    for i in range(1, n_months + 1):
        mois_idx = n + i - 1
        prev_annee = monthly["ANNEE"].iloc[-1]
        prev_mois  = monthly["MOIS"].iloc[-1]
        next_mois  = (prev_mois % 12) + i
        next_annee = prev_annee + (prev_mois + i - 1) // 12

        rev_pred  = max(0, np.polyval(rev_trend, mois_idx))
        cout_pred = max(0, np.polyval(cout_trend, mois_idx))

        input_d = {
            "REVENUS": rev_pred, "COUTS": cout_pred,
            "MOIS": next_mois % 12 or 12,
        }
        if "LOAD_FACTOR" in monthly.columns:
            input_d["LOAD_FACTOR"] = monthly["LOAD_FACTOR"].mean()
        pred = predict_single(input_d, model=model, scaler=scaler, feature_names=feature_names)

        rows.append({
            "ANNEE":        next_annee,
            "MOIS":         next_mois % 12 or 12,
            "REVENUS_PRED": rev_pred,
            "COUTS_PRED":   cout_pred,
            "PROFIT_PRED":  rev_pred - cout_pred,
            "PROBABILITE":  pred["probabilite"],
            "LABEL":        pred["label"],
        })

    return pd.DataFrame(rows)
