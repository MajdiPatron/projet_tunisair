import joblib
import os
import numpy as np

model_path = r"e:\rania cherif\APP tunisiar\model\model_best.pkl"
results_path = r"e:\rania cherif\APP tunisiar\model\results.json"

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"Model Type: {type(model)}")
    if hasattr(model, "coef_"):
        import json
        with open(results_path) as f:
            results = json.load(f)
        features = results.get("feature_names", [])
        coefs = model.coef_[0]
        intercept = model.intercept_[0]
        
        sorted_indices = np.argsort(np.abs(coefs))[::-1]
        print("\nTop Features by Coefficient Magnitude:")
        for i in sorted_indices:
            print(f"{features[i]}: {coefs[i]:.4f}")
        print(f"\nIntercept: {intercept:.4f}")
        print(f"Classes: {model.classes_}")
