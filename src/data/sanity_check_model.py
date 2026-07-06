import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# =========================================================
# LOAD DATA
# =========================================================

BASE_DIR = "src/data/processed/region_tensors"
manifest_path = os.path.join(BASE_DIR, "manifest.csv")

df = pd.read_csv(manifest_path)

print("\n==============================")
print("TOTAL PATIENTS:", len(df))
print("==============================")

# =========================================================
# BUILD SIMPLE FEATURES FROM .npy
# compress region matrix -> patient vector)
# =========================================================

X = []
y = []

for _, row in df.iterrows():
    path = os.path.join(BASE_DIR, row["filename"])
    arr = np.load(path)   # shape: (num_regions, 16)

    # -----------------------------
    # Patient-level summary features
    # -----------------------------
    features = [
        np.mean(arr),
        np.std(arr),
        np.min(arr),
        np.max(arr),

        np.mean(arr[:, 0]),  # num fragments per region
        np.mean(arr[:, 1]),  # mean length
        np.mean(arr[:, 4]),  # mean CpG
        np.mean(arr[:, 8]),  # methyl ratio
        np.mean(arr[:, 9]),  # entropy

        np.std(arr[:, 1]),
        np.std(arr[:, 4]),
        np.std(arr[:, 8]),
    ]

    X.append(features)
    y.append(row["label"])

X = np.array(X)
y = np.array(y)

print("\nFeature matrix shape:", X.shape)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# MODEL 1: LOGISTIC REGRESSION
# =========================================================

logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)

print("\n==============================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================")
print("Accuracy:", accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

# =========================================================
# MODEL 2: RANDOM FOREST
# =========================================================

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("\n==============================")
print("RANDOM FOREST RESULTS")
print("==============================")
print("Accuracy:", accuracy_score(y_test, pred_rf))
print(classification_report(y_test, pred_rf))

# =========================================================
# CONFUSION MATRIX PLOT
# =========================================================

cm = confusion_matrix(y_test, pred_rf)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importances = rf.feature_importances_

plt.figure()
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()