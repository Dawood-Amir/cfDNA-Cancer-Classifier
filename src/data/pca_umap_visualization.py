import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

BASE_DIR = "src/data/processed/region_tensors"
manifest = pd.read_csv(os.path.join(BASE_DIR, "manifest.csv"))

X = []
y = []

print("Extracting high-fidelity profile across all regional distributions...")

for _, row in manifest.iterrows():
    matrix = np.load(os.path.join(BASE_DIR, row["filename"]))  # Shape: (num_regions, 16)
    
    # Capture the full distribution shape of your features across regions
    patient_features = []
    for col_idx in range(matrix.shape[1]):
        col_data = matrix[:, col_idx]
        
        # Extract rich descriptive stats per feature column across all regions
        patient_features.extend([
            np.mean(col_data),
            np.std(col_data),
            np.percentile(col_data, 25),
            np.percentile(col_data, 50),
            np.percentile(col_data, 75),
            np.max(col_data)
        ])
        
    X.append(patient_features)
    y.append(row["label"])

X = np.array(X)
y = np.array(y)
print(f"New High-Fidelity Feature Matrix Shape: {X.shape}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", s=15, alpha=0.7)
plt.title("High-Fidelity PCA (No Aggressive Flattening)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.colorbar(scatter, ticks=[0, 1, 2, 3], label="Labels")
plt.show()

# --- UMAP ---
print("Running UMAP on high-fidelity features...")
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="tab10", s=15, alpha=0.7)
plt.title("High-Fidelity UMAP (No Aggressive Flattening)")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.colorbar(scatter, ticks=[0, 1, 2, 3], label="Labels")
plt.show()