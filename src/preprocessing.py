"""
TUNISAIR - Pipeline Preprocessing CRISP-DM Phase 3
Auteur: Data Science Team
"""
import pandas as pd
import numpy as np
import os, warnings
from scipy import stats
warnings.filterwarnings("ignore")

EXCEL_DIR = os.path.join(os.path.dirname(__file__), "..", "Modele_dimensionnel")
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")

# ─── 1. CHARGEMENT ───────────────────────────────────────────────────────────
def load_excel_datasets(data_dir=EXCEL_DIR):
    datasets, fichiers = {}, {"AVION":"AVION.xlsx","LIGNE":"LIGNE.xlsx",
                               "SOURCE":"SOURCE.xlsx","TEMPS":"TEMPS.xlsx","VOL":"VOL.xlsx"}
    for nom, f in fichiers.items():
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            df = pd.read_excel(p, engine="openpyxl")
            df.columns = [c.strip().upper().replace(" ","_") for c in df.columns]
            datasets[nom] = df
            print(f"  ✅ {nom}: {df.shape}")
        else:
            print(f"  ❌ {f} manquant")
    return datasets

def get_dataset_info(datasets):
    rows = []
    for n, df in datasets.items():
        rows.append({"Dataset":n,"Lignes":df.shape[0],"Colonnes":df.shape[1],
                     "Manquantes":df.isnull().sum().sum(),"Doublons":df.duplicated().sum()})
    return pd.DataFrame(rows)

# ─── 2. JOINTURE ─────────────────────────────────────────────────────────────
def _common_key(df1, df2, candidates):
    c1, c2 = set(df1.columns), set(df2.columns)
    for k in candidates:
        if k in c1 and k in c2: return k
    common = c1 & c2
    return list(common)[0] if common else None

def merge_datasets(datasets):
    vol = datasets.get("VOL", pd.DataFrame()).copy()
    joins = [
        ("AVION",  ["ID_AVION","AVION_ID","CODE_AVION","NUM_AVION"]),
        ("LIGNE",  ["ID_LIGNE","LIGNE_ID","CODE_LIGNE","NUM_LIGNE"]),
        ("TEMPS",  ["ID_TEMPS","TEMPS_ID","DATE_ID","MOIS_ID"]),
        ("SOURCE", ["ID_SOURCE","SOURCE_ID","CODE_SOURCE"]),
    ]
    df = vol.copy()
    for name, candidates in joins:
        dim = datasets.get(name, pd.DataFrame())
        if dim.empty: continue
        key = _common_key(df, dim, candidates)
        if key:
            df = df.merge(dim, on=key, how="left", suffixes=("", f"_{name}"))
            print(f"  🔗 VOL ↔ {name} via [{key}]")
    print(f"  ✅ Consolidé: {df.shape}")
    return df

# ─── 3. NETTOYAGE ────────────────────────────────────────────────────────────
def clean_data(df):
    n0 = len(df)
    df = df.drop_duplicates()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object","category"]).columns
    for c in num_cols:
        if df[c].isnull().any(): df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isnull().any():
            m = df[c].mode()
            df[c] = df[c].fillna(m[0] if not m.empty else "INCONNU")
    fin_cols = [c for c in num_cols if any(k in c for k in ["REVENU","COUT","PROFIT","SALES","FUEL","FRET"])]
    removed = 0
    for c in fin_cols:
        if c in df.columns and df[c].std() > 0:
            mask = np.abs(stats.zscore(df[c].fillna(df[c].median()))) < 4
            before = len(df)
            df = df[mask]
            removed += before - len(df)
    print(f"  ✅ Nettoyé: {n0}→{len(df)} lignes, outliers supprimés: {removed}")
    return df.reset_index(drop=True)

# ─── 4. FEATURE ENGINEERING ──────────────────────────────────────────────────
def _find_cols(cols, kws):
    return [c for c in cols if any(k in c for k in kws)]

def engineer_features(df):
    df = df.copy()
    cols = df.columns.tolist()

    # REVENUS
    rev_src = _find_cols(cols, ["SALES","DUTY_FREE","DUTYFREE","FRET","CARGO"])
    if rev_src:
        df["REVENUS"] = df[rev_src].fillna(0).sum(axis=1)
    elif "REVENUS" not in cols:
        rc = next((c for c in cols if "REVENU" in c or "REVENUE" in c), None)
        df["REVENUS"] = df[rc] if rc else 0

    # COUTS
    cout_src = _find_cols(cols, ["FUEL","HANDLING","GDS","ROUTE","REDEVANCE","MAINTENANCE"])
    if cout_src:
        df["COUTS"] = df[cout_src].fillna(0).sum(axis=1)
    elif "COUTS" not in cols:
        cc = next((c for c in cols if "COUT" in c or "COST" in c), None)
        df["COUTS"] = df[cc] if cc else 0

    # PROFIT
    df["PROFIT"] = df["REVENUS"] - df["COUTS"]

    # PAX & CAPACITE
    pax_col = next((c for c in cols if "PAX" in c and "REV" not in c), None)
    cap_col = next((c for c in cols if "CAPAC" in c or "SIEGE" in c), None)
    dist_col = next((c for c in cols if "DIST" in c), None)
    if not pax_col: df["PAX"] = 100; pax_col = "PAX"

    # LOAD FACTOR
    if cap_col and cap_col in df.columns:
        df["LOAD_FACTOR"] = np.where(df[cap_col]>0, (df[pax_col]/df[cap_col]).clip(0,1), 0)
    elif "LOAD_FACTOR" not in df.columns:
        df["LOAD_FACTOR"] = 0.75

    # REV_PER_PAX
    df["REV_PER_PAX"] = np.where(df[pax_col]>0, df["REVENUS"]/df[pax_col], 0)

    # COST_PER_KM
    if dist_col and dist_col in df.columns:
        df["COST_PER_KM"] = np.where(df[dist_col]>0, df["COUTS"]/df[dist_col], 0)
    else:
        df["COST_PER_KM"] = 0

    # TEMPOREL
    mois_col = next((c for c in cols if "MOIS" in c or "MONTH" in c), None)
    if mois_col:
        df["MOIS"] = pd.to_numeric(df[mois_col], errors="coerce").fillna(0)
        df["HAUTE_SAISON"] = df["MOIS"].isin([6,7,8,12]).astype(int)

    # RATIOS
    df["MARGE_OP"] = np.where(df["REVENUS"]>0, df["PROFIT"]/df["REVENUS"], 0)
    df["RATIO_COUT_REVENU"] = np.where(df["REVENUS"]>0, df["COUTS"]/df["REVENUS"], 0)

    # CIBLE
    df["RENTABLE"] = (df["PROFIT"]>0).astype(int)
    r = df["RENTABLE"].mean()*100
    print(f"  ✅ RENTABLE: {df['RENTABLE'].sum()} positifs ({r:.1f}%), {(df['RENTABLE']==0).sum()} négatifs")
    return df

# ─── 5. ENCODAGE ─────────────────────────────────────────────────────────────
def encode_and_normalize(df):
    df = df.copy()
    cat_cols = [c for c in df.select_dtypes(include=["object","category"]).columns
                if not any(c.startswith(p) for p in ["ID_","CODE_","NUM_","DATE","LIB"])
                and df[c].nunique() < 50]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
    drop_cols = [c for c in df.columns if any(c.startswith(p) for p in ["ID_","CODE_"])]
    df = df.drop(columns=drop_cols, errors="ignore")
    date_cols = df.select_dtypes(include=["datetime64","object"]).columns.tolist()
    df = df.drop(columns=date_cols, errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"  ✅ Encodage final: {df.shape}")
    return df

# ─── PIPELINE COMPLET ─────────────────────────────────────────────────────────
def run_preprocessing_pipeline(data_dir=EXCEL_DIR):
    print("="*60+"\n  TUNISAIR — PIPELINE PREPROCESSING\n"+"="*60)
    datasets = load_excel_datasets(data_dir)
    print(f"\n{get_dataset_info(datasets).to_string()}")
    df_merged = merge_datasets(datasets)
    df_clean  = clean_data(df_merged)
    df_feat   = engineer_features(df_clean)
    df_final  = encode_and_normalize(df_feat)
    os.makedirs(DATA_DIR, exist_ok=True)
    df_final.to_csv(os.path.join(DATA_DIR,"dataset_final.csv"), index=False)
    df_feat.to_csv(os.path.join(DATA_DIR,"dataset_vis.csv"), index=False)
    print(f"\n💾 Sauvegardé dans {DATA_DIR}")
    return df_final

if __name__ == "__main__":
    df = run_preprocessing_pipeline()
    print(df.shape, df["RENTABLE"].value_counts())
