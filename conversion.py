import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
import tempfile
import pygame
import time
import threading
from sklearn.preprocessing import StandardScaler
from gtts import gTTS  # ✅ Text-to-Speech for Kannada

# ✅ **Initialize MediaPipe Hand Tracking**
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ✅ **Set Paths**
BASE_PATH = r"C:\Users\Kingshuk Maji\Documents\Sign_Connect\Sign Connect"
MODEL_PATH = os.path.join(BASE_PATH, "Models", "SVM", "svm_model.pkl")
SCALER_PATH = os.path.join(BASE_PATH, "Models", "SVM", "scaler.pkl")
PCA_PATH = os.path.join(BASE_PATH, "Models", "SVM", "pca.pkl")

# ✅ **Function to Extract Hand Landmarks**
def detect_hand_landmarks(frame):
    """Extracts hand landmarks from a frame and returns (1, 63) NumPy array."""
    if frame is None:
        print("❌ ERROR: Frame is empty!")
        return np.zeros((1, 63))  # Prevents crashes

    results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark])
        return landmarks.flatten().reshape(1, -1)  # Convert (21,3) → (1,63)

    return np.zeros((1, 63))  # Prevents crashes if no hand is detected

# ✅ **Improved Speech Output Function**
def speak_kannada(text):
    """Converts Kannada text to speech and plays the sound asynchronously."""
    if text and text.strip():  # ✅ Prevent empty speech
        print(f"🗣 Speaking: {text}")

        def play_audio():
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                    audio_path = temp_audio.name

                tts = gTTS(text=text, lang='kn')
                tts.save(audio_path)

                pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)  # ✅ Prevents UI freezing

                pygame.mixer.quit()
                time.sleep(0.5)  # Prevent premature file deletion
                os.remove(audio_path)
            except Exception as e:
                print(f"⚠️ Voice Output Error: {e}")

        threading.Thread(target=play_audio, daemon=True).start()

# ✅ **Real-Time Recognition Class**
class RealTimeRecognition:
    def __init__(self, model, scaler, pca=None):
        self.model = model
        self.scaler = scaler
        self.pca = pca
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.video_capture.isOpened():
            print("❌ ERROR: Webcam not accessible! Try restarting your system.")
            exit()

    def process_frame(self, frame):
        """Extracts features, scales them, applies PCA (if used), and predicts Kannada sign."""
        feature = detect_hand_landmarks(frame)

        # ✅ **Fix: Ensure PCA is applied safely**
        if self.pca:
            try:
                if feature.shape[1] == self.pca.n_components_:
                    feature = self.pca.transform(feature)
                else:
                    print("⚠️ PCA: Feature size mismatch. Skipping PCA transformation.")
            except Exception as e:
                print(f"⚠️ PCA Transformation Error: {e}")
                return "None", "None"

        # ✅ **Fix: Handle feature size safely**
        try:
            if feature.shape[1] != self.scaler.n_features_in_:
                print("⚠️ Scaling Error: Feature shape mismatch. Adjusting feature shape...")
                feature = np.zeros((1, self.scaler.n_features_in_))  # Fix shape
            
            feature = self.scaler.transform(feature)
            prediction = self.model.predict(feature)
            predicted_sign = prediction[0] if prediction else "None"

            # ✅ **Fix: Kannada Prediction is Direct**
            kannada_sign = predicted_sign  

            print(f"🔮 Recognized Sign: {predicted_sign}")

            # ✅ **Fix: Speak the Kannada sign**
            if kannada_sign and kannada_sign != "None":
                speak_kannada(kannada_sign)

            return predicted_sign, kannada_sign
        except Exception as e:
            print(f"❌ Model Prediction Error: {e}")
            return "None", "None"

    def run_webcam(self):
        """Runs real-time recognition using webcam."""
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                print("❌ ERROR: Cannot read webcam frame!")
                break

            frame = cv2.flip(frame, 1)
            predicted_sign, kannada_sign = self.process_frame(frame)

            # ✅ **Fix: Ensure correct Kannada sign is displayed**
            if not kannada_sign or kannada_sign == "None":
                kannada_sign = "No Sign Detected"

            # Display prediction
            cv2.putText(frame, f"Prediction: {kannada_sign}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Real-Time Hand Sign Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()

# ✅ **Run the Steps**
if __name__ == "__main__":
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Model and Scaler loaded successfully!")

        pca = joblib.load(PCA_PATH) if os.path.exists(PCA_PATH) else None
        if pca:
            print("✅ PCA Model loaded successfully!")

        recognizer = RealTimeRecognition(model, scaler, pca)
        recognizer.run_webcam()
    else:
        print("❌ ERROR: Model or Scaler file is missing!")