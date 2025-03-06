from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ✅ Create an SVM model with scaling
svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
