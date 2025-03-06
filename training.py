import os
import numpy as np
import joblib
import cv2
from kivy.clock import Clock
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from kivy.uix.screenmanager import Screen
from ultralytics import YOLO
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput


class LearningScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ✅ **Main Layout**
        layout = BoxLayout(orientation="vertical", spacing=20, padding=30)

        # ✅ **Title Label**
        title_label = Label(text="Learning & Training", font_size=28, bold=True, size_hint=(1, 0.2))
        layout.add_widget(title_label)

        # ✅ **Label Input for Training**
        self.label_input = TextInput(hint_text="Enter label for training", size_hint=(1, 0.15), multiline=False)
        layout.add_widget(self.label_input)

        # ✅ **Start Learning Button**
        start_learning_button = Button(text="Start Learning", size_hint=(1, 0.15))
        start_learning_button.bind(on_press=self.start_learning)
        layout.add_widget(start_learning_button)

        # ✅ **Back Button**
        back_button = Button(text="Back", size_hint=(1, 0.15))
        back_button.bind(on_press=self.back_to_second)
        layout.add_widget(back_button)

        # ✅ **Add Layout to Screen**
        self.add_widget(layout)

    def back_to_second(self, instance):
        """Safely navigate back to the second screen."""
        if "second" in self.manager.screen_names:
            Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'second'), 0)
        else:
            print("❌ Error: 'second' screen not found in ScreenManager.")

    def start_learning(self, instance):
        """Handles YOLO feature extraction and SVM training."""
        label_name = self.label_input.text.strip()
        if not label_name:
            print("❌ Error: Please enter a label before training!")
            return

        # ✅ **File Paths**
        BASE_PATH = r"C:\Users\Kingshuk Maji\Documents\Sign_Connect\Sign Connect"
        DATA_PATH = os.path.join(BASE_PATH, "Datasets")
        YOLO_MODEL_PATH = os.path.join(BASE_PATH, "Models", "Yolo", "yolov8_model.pt")
        YOLO_FEATURES_SAVE_PATH = os.path.join(BASE_PATH, "Models", "Yolo", "yolov8_model.pkl")
        SVM_MODEL_SAVE_PATH = os.path.join(BASE_PATH, "Models", "SVM", "svm_model.pkl")

        # ✅ **Check if YOLO Model Exists**
        if not os.path.exists(YOLO_MODEL_PATH):
            print("❌ Error: YOLO model file not found!")
            return

        # ✅ **Load YOLO Model**
        yolo_model = YOLO(YOLO_MODEL_PATH)

        # ✅ **Check if Dataset Exists**
        if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
            print("❌ Error: No dataset found in 'Datasets' directory!")
            return

        # ✅ **Extract Features using YOLO**
        features, labels = [], []
        CLASSES_LIST = os.listdir(DATA_PATH)

        print("📂 **Classes Found:**", CLASSES_LIST)

        for label_id, sign_name in enumerate(CLASSES_LIST):
            sign_path = os.path.join(DATA_PATH, sign_name)
            if not os.path.isdir(sign_path):
                continue  # Skip if not a directory

            for filename in os.listdir(sign_path):
                image_path = os.path.join(sign_path, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"⚠️ Warning: Could not read {image_path}. Skipping.")
                    continue

                # ✅ **Perform YOLO inference**
                results = yolo_model.predict(image)

                # ✅ **Extract YOLO bounding box features**
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # ✅ **Ensure bounding box is within image dimensions**
                        h, w, _ = image.shape
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                        crop = image[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue  # Skip invalid crops

                        resized_crop = cv2.resize(crop, (64, 64))
                        feature_vector = resized_crop.flatten()  # ✅ Convert image to 1D feature vector

                        features.append(feature_vector)
                        labels.append(label_id)

        # ✅ **Convert Features to NumPy Arrays**
        features = np.array(features)
        labels = np.array(labels)

        # ✅ **Save Extracted Features**
        joblib.dump((features, labels), YOLO_FEATURES_SAVE_PATH)
        print(f"\n✅ YOLO features saved at: {YOLO_FEATURES_SAVE_PATH}")

        # ✅ **Train SVM Model**
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)

        # ✅ **Split Dataset**
        X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

        # ✅ **Train SVM**
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)

        # ✅ **Save SVM Model**
        joblib.dump(svm_model, SVM_MODEL_SAVE_PATH)
        print(f"\n✅ SVM model saved at: {SVM_MODEL_SAVE_PATH}")

        # ✅ **Evaluate Model**
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✅ Overall Accuracy: {accuracy * 100:.2f}%")

        # ✅ **Display Confusion Matrix**
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.savefig('confusion_matrix.png')
       
