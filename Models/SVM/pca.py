import os
import joblib
import numpy as np
from sklearn.decomposition import PCA

# ✅ **Set Paths**
FEATURES_PATH = r"C:\Users\Kingshuk Maji\Documents\Sign_Connect\Sign Connect\Features"
PCA_PATH = r"C:\Users\Kingshuk Maji\Documents\Sign_Connect\Sign Connect\Models\SVM\pca.pkl"

# ✅ **Load Features from `.npy` Files**
feature_list = []
for file in os.listdir(FEATURES_PATH):
    if file.endswith(".npy"):
        file_path = os.path.join(FEATURES_PATH, file)
        features = np.load(file_path)

        # ✅ **Ensure Features are 2D**
        if len(features.shape) == 3:  
            features = features.reshape(features.shape[0], -1)  # Convert (N, 21, 3) → (N, 63)

        elif features.shape[1] == 12288:  
            print(f"⚠️ {file} has image pixel data! Skipping.")  
            continue  # Skip image-based feature files

        elif features.shape[1] != 63:  
            print(f"⚠️ {file} has unexpected shape {features.shape}. Skipping.")  
            continue  # Skip files with incorrect feature dimensions

        feature_list.append(features)

# ✅ **Stack Features**
if feature_list:
    X = np.vstack(feature_list)  # (N, 63)
    print(f"🔍 Features Shape Before PCA: {X.shape}")
else:
    print("❌ No valid feature files found!")
    exit()

# ✅ **Apply PCA (Reduce from 63 → 63 for consistency)**
pca = PCA(n_components=63)
X_reduced = pca.fit_transform(X)
print(f"✅ Features Shape After PCA: {X_reduced.shape}")

# ✅ **Save PCA Model**
joblib.dump(pca, PCA_PATH)
print(f"🎉 PCA model saved at: {PCA_PATH}")
