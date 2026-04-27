"""
TUNISAIR - Module d'Entraînement des Modèles ML
CRISP-DM Phase 4: Modeling + Phase 5: Evaluation
Modèles : Logistic Regression | Random Forest | XGBoost
"""
import pandas as pd
import numpy as np
import os, json, warnings, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "model")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def _ensure_dirs():
    for d in [MODEL_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)


# ─── DÉFINITION DES MODÈLES ──────────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42,
                                                       class_weight="balanced", n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=200, random_state=42,
                                              use_label_encoder=False, eval_metric="logloss",
                                              scale_pos_weight=1),
    }


# ─── TUNING HYPERPARAMÈTRES ──────────────────────────────────────────────────
def tune_xgboost(X_train, y_train, n_iter=20, cv=5):
    """RandomizedSearchCV sur XGBoost."""
    param_dist = {
        "n_estimators":    [100, 200, 300, 500],
        "max_depth":       [3, 4, 5, 6, 7],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "subsample":       [0.6, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0],
        "min_child_weight":[1, 3, 5],
        "gamma":           [0, 0.1, 0.2],
    }
    xgb = XGBClassifier(random_state=42, use_label_encoder=False,
                         eval_metric="logloss", n_jobs=-1)
    search = RandomizedSearchCV(
        xgb, param_dist, n_iter=n_iter, cv=cv, scoring="roc_auc",
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"  🎯 Meilleurs params XGBoost: {search.best_params_}")
    print(f"  🎯 Meilleur ROC-AUC CV: {search.best_score_:.4f}")
    return search.best_estimator_


# ─── ÉVALUATION D'UN MODÈLE ──────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Calcule toutes les métriques d'évaluation."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model":     model_name,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


# ─── CROSS-VALIDATION ────────────────────────────────────────────────────────
def cross_validate_model(model, X, y, cv=5) -> dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = {}
    for metric in ["accuracy", "f1", "roc_auc"]:
        sc = cross_val_score(model, X, y, cv=skf, scoring=metric, n_jobs=-1)
        scores[metric] = {"mean": sc.mean(), "std": sc.std(), "all": sc.tolist()}
    return scores


# ─── VISUALISATIONS ──────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, model_name: str, save_dir: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Non Rentable","Rentable"],
                yticklabels=["Non Rentable","Rentable"], ax=ax)
    ax.set_title(f"Matrice de Confusion — {model_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Réel"); ax.set_xlabel("Prédit")
    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_roc_curves(models_data: list, X_test, y_test, save_dir: str):
    """models_data = [(name, model), ...]"""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#E30613", "#1C3F6E", "#2CA02C"]
    for (name, model), color in zip(models_data, colors):
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"--", color="gray", lw=1)
    ax.set_xlabel("Taux Faux Positifs"); ax.set_ylabel("Taux Vrais Positifs")
    ax.set_title("Courbes ROC — Comparaison Modèles", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curves_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_feature_importance(model, feature_names: list, model_name: str, save_dir: str):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    fi = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    top_n = fi.tail(min(20, len(fi)))

    fig, ax = plt.subplots(figsize=(8, max(4, len(top_n)*0.35)))
    bars = ax.barh(top_n.index, top_n.values, color="#E30613", alpha=0.85)
    ax.set_xlabel("Importance"); ax.set_title(f"Feature Importance — {model_name}", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"feature_importance_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_models_comparison(all_metrics: list, save_dir: str):
    df = pd.DataFrame(all_metrics).set_index("model")
    metrics_to_plot = ["accuracy","precision","recall","f1","roc_auc"]
    df_plot = df[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    colors = ["#E30613","#1C3F6E","#2CA02C"]
    for i, (model_name, row) in enumerate(df_plot.iterrows()):
        ax.bar(x + i*width, row.values, width, label=model_name, color=colors[i], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot], fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Comparaison des Modèles — Toutes Métriques", fontweight="bold", fontsize=12)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "models_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ─── PIPELINE COMPLET D'ENTRAÎNEMENT ─────────────────────────────────────────
def run_training_pipeline(split_data: dict, feature_names: list,
                           tune: bool = True) -> dict:
    """
    Pipeline complet : entraîne, évalue et sauvegarde les 3 modèles.

    Args:
        split_data: dict retourné par features.split_and_scale()
        feature_names: liste des noms de features
        tune: si True, applique RandomizedSearchCV sur XGBoost

    Returns:
        dict avec modèles entraînés, métriques et chemins des graphiques
    """
    _ensure_dirs()

    X_train_s = split_data["X_train_scaled"]
    X_test_s  = split_data["X_test_scaled"]
    X_train   = split_data["X_train"]
    X_test    = split_data["X_test"]
    y_train   = split_data["y_train"]
    y_test    = split_data["y_test"]

    print("="*60)
    print("  TUNISAIR — PIPELINE ENTRAÎNEMENT ML")
    print("="*60)

    # ── SMOTE si déséquilibre ────────────────────────────────────────────────
    ratio = y_train.mean()
    if ratio < 0.35 or ratio > 0.65:
        print(f"\n⚠️  Déséquilibre détecté ({ratio:.1%}) — Application SMOTE")
        sm = SMOTE(random_state=42)
        X_train_s_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)
        X_train_bal, _             = sm.fit_resample(X_train, y_train)
        print(f"  ✅ Après SMOTE: {len(X_train_s_bal)} échantillons")
    else:
        X_train_s_bal, y_train_bal = X_train_s, y_train
        X_train_bal = X_train

    # ── Tuning XGBoost ───────────────────────────────────────────────────────
    models = get_models()
    if tune:
        print("\n🔧 Tuning XGBoost (RandomizedSearchCV)...")
        models["XGBoost"] = tune_xgboost(X_train_bal, y_train_bal)

    # ── Entraînement ─────────────────────────────────────────────────────────
    trained_models, all_metrics, cv_results = {}, [], {}

    for name, model in models.items():
        print(f"\n📈 Entraînement : {name}")
        # LR utilise données scalées; RF et XGBoost peuvent utiliser brutes
        if name == "Logistic Regression":
            model.fit(X_train_s_bal, y_train_bal)
            X_eval = X_test_s
        else:
            model.fit(X_train_bal, y_train_bal)
            X_eval = X_test

        # Évaluation
        metrics = evaluate_model(model, X_eval, y_test, name)
        all_metrics.append(metrics)
        trained_models[name] = (model, X_eval)

        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-Score : {metrics['f1']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

        # Cross-validation
        cv_data  = cross_validate_model(model, X_train_bal, y_train_bal)
        cv_results[name] = cv_data
        print(f"  CV ROC-AUC: {cv_data['roc_auc']['mean']:.4f} ± {cv_data['roc_auc']['std']:.4f}")

        # Matrice de confusion
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, name, REPORT_DIR)

        # Feature Importance
        plot_feature_importance(model, feature_names, name, REPORT_DIR)

    # ── ROC comparaison ──────────────────────────────────────────────────────
    roc_models = [(n, m) for n, (m, _) in trained_models.items()]
    # Utiliser X_test_s pour LR, X_test pour les autres dans la comparaison ROC
    roc_path = plot_roc_curves(
        [(n, m) for n,(m,_) in trained_models.items()],
        X_test_s, y_test, REPORT_DIR
    )

    # ── Comparaison globale ──────────────────────────────────────────────────
    comp_path = plot_models_comparison(all_metrics, REPORT_DIR)

    # ── Sélection meilleur modèle (ROC-AUC) ─────────────────────────────────
    best_metrics = max(all_metrics, key=lambda m: m["roc_auc"])
    best_name    = best_metrics["model"]
    best_model   = trained_models[best_name][0]
    print(f"\n🏆 MEILLEUR MODÈLE : {best_name} (ROC-AUC={best_metrics['roc_auc']:.4f})")

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "model_best.pkl")
    joblib.dump(best_model, model_path)

    # Sauvegarde de tous les modèles
    for name, (model, _) in trained_models.items():
        p = os.path.join(MODEL_DIR, f"model_{name.replace(' ','_').lower()}.pkl")
        joblib.dump(model, p)

    # Sauvegarde métriques JSON
    results = {
        "best_model": best_name,
        "metrics":    all_metrics,
        "cv_results": cv_results,
        "feature_names": feature_names,
    }
    with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Modèle sauvegardé : {model_path}")
    print("="*60+"\n  ✅ PIPELINE ENTRAÎNEMENT TERMINÉ\n"+"="*60)

    return {
        "models":       {n: m for n,(m,_) in trained_models.items()},
        "best_model":   best_model,
        "best_name":    best_name,
        "all_metrics":  all_metrics,
        "cv_results":   cv_results,
        "feature_names":feature_names,
    }
