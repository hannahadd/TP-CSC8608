import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

# 1. Chargement et Séparation des données
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0: Maligne, 1: Bénigne

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Entraînement d'une "Boîte Noire" (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  

print(f"Accuracy du Random Forest : {model.score(X_test, y_test):.4f}")

# 3. Explicabilité Post-Hoc avec SHAP
explainer = shap.TreeExplainer(model)  
shap_values = explainer(X_test)        

# Classe 1 (Bénigne) pour les visualisations
shap_values_class1 = shap_values[:, :, 1]

# ---- BONUS: imprime ce qu'il faut pour répondre au rapport ----
vals = shap_values_class1.values  # (n_samples, n_features)
mean_abs = np.mean(np.abs(vals), axis=0)
top_idx = np.argsort(mean_abs)[::-1][:3]

print("\nTop-3 features (global, mean |SHAP|) — classe 1 (Bénigne):")
for i in top_idx:
    print(f"- {X_test.columns[i]} : {mean_abs[i]:.6f}")

patient_idx = 0
sv0 = shap_values_class1[patient_idx]
local_vals = sv0.values
best_i = int(np.argmax(np.abs(local_vals)))
best_feat = X_test.columns[best_i]
best_feat_value = float(X_test.iloc[patient_idx, best_i])
best_contrib = float(local_vals[best_i])

print(f"\nPatient {patient_idx} — plus grosse contribution (|SHAP|):")
print(f"- Feature: {best_feat}")
print(f"- Valeur exacte (X_test): {best_feat_value}")
print(f"- SHAP (classe 1): {best_contrib:.6f}")

# 4. Explicabilité Locale : Waterfall Plot (Un seul patient)
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values_class1[patient_idx], show=False)
plt.title(f"Explication Locale SHAP - Patient {patient_idx}")
plt.tight_layout()
output_local = "shap_waterfall.png"
plt.savefig(output_local)
plt.close()
print(f"\nWaterfall plot sauvegardé dans {output_local}")

# 5. Explicabilité Globale : Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_class1, X_test, show=False)
plt.title("Importance globale et directionnelle des variables (SHAP)")
plt.tight_layout()
output_global = "shap_summary.png"
plt.savefig(output_global)
plt.close()
print(f"Summary plot sauvegardé dans {output_global}")