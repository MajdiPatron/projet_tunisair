# 🛫 TUNISAIR — Système de Prédiction de Rentabilité des Lignes

> **Solution IA End-to-End** — CRISP-DM · XGBoost · Streamlit Premium

---

## 📁 Structure du Projet

```
APP tunisiar/
│
├── Modele_dimensionnel/        ← Datasets Excel sources
│   ├── AVION.xlsx
│   ├── LIGNE.xlsx
│   ├── SOURCE.xlsx
│   ├── TEMPS.xlsx
│   └── VOL.xlsx
│
├── src/                        ← Modules Python
│   ├── __init__.py
│   ├── preprocessing.py        ← CRISP-DM Phase 3 : Data Prep
│   ├── features.py             ← Feature selection & split
│   ├── train.py                ← CRISP-DM Phase 4-5 : ML + Éval
│   └── predict.py              ← CRISP-DM Phase 6 : Déploiement + SHAP
│
├── app/                        ← Application Streamlit
│   ├── __init__.py
│   ├── streamlit_app.py        ← Point d'entrée principal
│   ├── styles.py               ← CSS + composants HTML
│   ├── page_dashboard.py       ← Dashboard KPIs + graphiques
│   ├── page_prediction.py      ← Prédiction + What-If + Forecast
│   └── page_models.py          ← Évaluation + SHAP + Matrices
│
├── data/                       ← Données générées (auto)
│   ├── dataset_final.csv
│   └── dataset_vis.csv
│
├── model/                      ← Modèles sauvegardés (auto)
│   ├── model_best.pkl
│   ├── model_random_forest.pkl
│   ├── model_logistic_regression.pkl
│   ├── model_xgboost.pkl
│   ├── scaler.pkl
│   └── results.json
│
├── reports/                    ← Graphiques générés (auto)
│   ├── roc_curves_comparison.png
│   ├── models_comparison.png
│   ├── confusion_*.png
│   ├── feature_importance_*.png
│   └── shap_summary.png
│
├── .streamlit/
│   └── config.toml             ← Thème dark Tunisair
│
├── run_training.py             ← Script pipeline complet
├── requirements.txt
└── README.md
```

---

## ⚡ Installation & Exécution

### Étape 1 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 2 — Entraîner les modèles

```bash
python run_training.py
```

Ce script exécute automatiquement :
1. ✅ Chargement & jointure des 5 datasets Excel
2. ✅ Nettoyage (doublons, outliers, valeurs manquantes)
3. ✅ Feature Engineering (PROFIT, LOAD_FACTOR, REV_PER_PAX, COST_PER_KM…)
4. ✅ Entraînement de 3 modèles (LR, RF, XGBoost)
5. ✅ Tuning hyperparamètres (RandomizedSearchCV)
6. ✅ Évaluation complète (Accuracy, F1, ROC-AUC, Confusion Matrix)
7. ✅ Sauvegarde du meilleur modèle

### Étape 3 — Lancer l'application

```bash
streamlit run app/streamlit_app.py
```

> ℹ️ L'app fonctionne **sans entraînement préalable** grâce aux données de démo intégrées.

---

## 🎯 Variable Cible

```python
RENTABLE = 1  si  PROFIT > 0
RENTABLE = 0  sinon
```

## 📊 KPI Métiers

| Indicateur       | Formule                          |
|-----------------|----------------------------------|
| REVENUS         | SALES + DUTY_FREE + FRET         |
| COUTS           | FUEL + HANDLING + GDS + ROUTE    |
| PROFIT          | REVENUS − COUTS                  |
| LOAD_FACTOR     | PAX / CAPACITE                   |
| REV_PER_PAX     | REVENUS / PAX                    |
| COST_PER_KM     | COUTS / DISTANCE                 |
| MARGE_OP        | PROFIT / REVENUS                 |

## 🤖 Modèles ML

| Modèle               | Avantage                        |
|---------------------|---------------------------------|
| Logistic Regression | Interprétable, baseline solide  |
| Random Forest       | Robuste, feature importance     |
| XGBoost ⭐           | Meilleure performance, SHAP     |

## 🌐 Pages de l'Application

- **🏠 Dashboard** — KPIs temps réel, distribution profits, saisonnalité
- **🔮 Prédiction** — Formulaire IA, jauge probabilité, scénarios what-if, forecast
- **📊 Modèles** — Comparaison métriques, matrices de confusion, SHAP
- **📋 Données** — Explorateur interactif, stats descriptives

---

## 🛠️ Technologies

`Python 3.10+` · `Streamlit 1.33` · `XGBoost` · `scikit-learn` · `SHAP` · `Plotly` · `Pandas`
# projet_tunisair
