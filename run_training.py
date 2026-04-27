"""
TUNISAIR - Script Principal d'Entrainement
Lance le pipeline complet CRISP-DM
Usage: python run_training.py
"""
import sys, os
# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import run_preprocessing_pipeline
from src.features import select_features, split_and_scale
from src.train import run_training_pipeline

def main():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  TUNISAIR -- SYSTEME PREDICTION RENTABILITE")
    print("  Pipeline CRISP-DM Complet")
    print(f"{sep}\n")

    # Phase 3 : Preprocessing
    df_final = run_preprocessing_pipeline()

    # Phase 4a : Features
    X, y, feature_names = select_features(df_final)

    # Phase 4b : Split & Scale
    split_data = split_and_scale(X, y)

    # Phase 4-5 : Training & Evaluation
    results = run_training_pipeline(split_data, feature_names, tune=True)

    print(f"\n[RESUME FINAL]")
    print(f"  Meilleur modele : {results['best_name']}")
    for m in results["all_metrics"]:
        print(f"\n  [{m['model']}]")
        print(f"    Accuracy : {m['accuracy']:.4f}")
        print(f"    F1-Score : {m['f1']:.4f}")
        print(f"    ROC-AUC  : {m['roc_auc']:.4f}")

    print("\n[OK] Pipeline termine!")
    print("     Lancez l'app avec : streamlit run app/streamlit_app.py\n")

if __name__ == "__main__":
    main()

