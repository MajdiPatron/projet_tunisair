"""
TUNISAIR - Feature Selection & Préparation pour Modélisation
CRISP-DM Phase 3 (suite) et Phase 4 (Modeling)
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")

# ─── FEATURES MÉTIERS PRIORITAIRES ───────────────────────────────────────────
CORE_FEATURES = [
    "REVENUS", "COUTS", "LOAD_FACTOR", "REV_PER_PAX",
    "COST_PER_KM", "MARGE_OP", "RATIO_COUT_REVENU",
]
OPTIONAL_FEATURES = [
    "HAUTE_SAISON", "MOIS", "ANNEE",
    "DISTANCE", "CAPACITE",
]

def select_features(df: pd.DataFrame, target: str = "RENTABLE") -> tuple:
    """
    Sélectionne les features disponibles dans le dataset et retourne
    X (features) et y (cible).

    Returns:
        (X: pd.DataFrame, y: pd.Series, feature_names: list)
    """
    all_candidates = CORE_FEATURES + OPTIONAL_FEATURES

    # Recherche souple des colonnes (les noms peuvent varier selon jointure)
    available = []
    for cand in all_candidates:
        # Cherche correspondance exacte ou partielle
        matches = [c for c in df.columns if cand in c and c != target]
        if matches:
            available.append(matches[0])

    # Ajoute colonnes booléennes one-hot automatiquement créées
    bool_cols = [c for c in df.columns
                 if c.startswith(("TYPE_","MARCHE_","PROPRIETAIRE_","CONTINENT_"))
                 and c not in available]
    available += bool_cols[:10]  # max 10 one-hot

    # Dédoublonnage
    available = list(dict.fromkeys(available))

    # S'assurer que target n'est pas dans features
    if target in available:
        available.remove(target)

    if not available:
        raise ValueError("Aucune feature utilisable trouvée dans le dataset.")

    X = df[available].copy()
    y = df[target].copy() if target in df.columns else pd.Series([0]*len(df))

    # Remplacement infinis et NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  ✅ {len(available)} features sélectionnées : {available}")
    print(f"  ✅ Distribution cible: {y.value_counts().to_dict()}")
    return X, y, available


def split_and_scale(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                    random_state: int = 42) -> dict:
    """
    Split train/test + StandardScaler.

    Returns:
        dict avec X_train, X_test, y_train, y_test, scaler, X_train_scaled, X_test_scaled
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    # Sauvegarder le scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"  ✅ Split: train={len(X_train)}, test={len(X_test)} | scaler sauvegardé")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
    }


def load_prepared_data(data_path: str = None):
    """Charge le dataset final et retourne X, y préparés."""
    if data_path is None:
        data_path = os.path.join(DATA_DIR, "dataset_final.csv")
    df = pd.read_csv(data_path)
    return select_features(df)
