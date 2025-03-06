import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ✅ **Paths**
FEATURES_PATH = r"C:\Users\Kingshuk Maji\Documents\Sign_Connect\Sign Connect\Features"
MODEL_PATH = r"C:\Users\Kingshuk Maji\Documents\Sign_Connect\Sign Connect\Models\SVM\svm_model.pkl"
SCALER_PATH = r"C:\Users\Kingshuk Maji\Documents\Sign_Connect\Sign Connect\Models\SVM\scaler.pkl"

# ✅ **Kannada Sign Dictionary** (English -> Kannada)
dic = {
    "Ba": "ಬ", "Bha": "ಭ", "Cha": "ಚ", "Chha": "ಛ", "Da": "ದ", "Daa": "ಡ",
    "Dhaa": "ಢ", "Dhha": "ಧ", "Ga": "ಗ", "Gha": "ಘ", "Ha": "ಹ", "Ja": "ಜ",
    "Jha": "ಝ", "Ka": "ಕ", "Kha": "ಖ", "la": "ಲ", "lla": "ಳ", "Ma": "ಮ",
    "Na": "ಞ", "Nah": "ನ", "Nna": "ಣ", "Nya": "ಙ", "Pa": "ಪ", "Pha": "ಫ",
    "Ra": "ರ", "sa": "ಸ", "sha": "ಷ", "she": "ಶ", "Ta": "ತ", "Taa": "ಟ",
    "Tha": "ಥ", "Thaa": "ಠ", "va": "ವ", "Ya": "ಯ"
}

# ✅ **Load Features & Labels**
features, labels = [], []
missing_features = []

print("\n🔍 Checking available feature files...\n")

for eng_label, kannada_label in dic.items():
    feature_file = os.path.join(FEATURES_PATH, f"{eng_label}.npy")

    if os.path.exists(feature_file):
        try:
            data = np.load(feature_file, allow_pickle=True)

            # ✅ **Ensure correct shape**
            if data.ndim == 3 and data.shape[1:] == (21, 3):
                data = data.reshape(data.shape[0], -1)  # Convert to (N, 63)
                print(f"✅ Reshaped {eng_label}.npy to {data.shape}.")
            elif data.ndim != 2 or data.shape[1] != 63:
                print(f"❌ Error: {eng_label}.npy has incorrect shape {data.shape}. Skipping.")
                continue

            features.extend(data)
            labels.extend([kannada_label] * len(data))  

        except Exception as e:
            print(f"❌ Error loading {feature_file}: {e}")
            missing_features.append(eng_label)

    else:
        print(f"❌ Feature file missing for {eng_label} ({kannada_label})")
        missing_features.append(eng_label)

# ✅ **Convert to NumPy arrays**
if len(features) == 0 or len(labels) == 0:
    raise ValueError("\n❌ No valid features or labels found. Check `.npy` file structure.")

features = np.array(features)
labels = np.array(labels)

print(f"\n✅ Total Features Loaded: {features.shape}")
print(f"✅ Total Labels Loaded: {len(labels)}")

# ✅ **Check if feature & label count match**
if features.shape[0] != len(labels):
    raise ValueError(f"\n❌ Feature count {features.shape[0]} does not match label count {len(labels)}!")

# ✅ **Split Data into Training & Testing**
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)

# ✅ **Normalize Features**
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ **Debug: Print Scaled Feature Sample**
print("\n🔍 Training Sample Before Scaling:", features[0])
print("✅ Training Sample After Scaling:", X_train[0])

# ✅ **Train the SVM Model with GridSearch**
param_grid = {
    'C': [1, 10, 100],  # ✅ **Smaller Grid for Better Performance**
    'kernel': ['rbf']  # ✅ **Only RBF (Best for Hand Sign Recognition)**
}

print("\n🔍 Performing Grid Search for best SVM parameters...")
grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"\n✅ Best Hyperparameters: {best_params}")

# ✅ **Train Best SVM Model**
svm_model = SVC(**best_params, probability=True)
svm_model.fit(X_train, y_train)

# ✅ **Save Model & Scaler**
joblib.dump(svm_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n✅ Model and scaler saved successfully!")
print(f"📁 Model: {MODEL_PATH}")
print(f"📁 Scaler: {SCALER_PATH}")

# ✅ **Evaluate Model**
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Overall Accuracy: {accuracy * 100:.2f}%")

# ✅ **Confusion Matrix**
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=list(dic.values()), yticklabels=list(dic.values()))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')
plt.show()

# ✅ **Classification Report**
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=list(dic.values())))