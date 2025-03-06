from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import joblib
import numpy as np
import os
import threading
import pygame
import time
from gtts import gTTS
import queue  # Prevent overlapping speech
import tempfile
from kivy.app import App
from kivy.core.text import LabelBase  
from conversion import RealTimeRecognition
from sklearn.preprocessing import StandardScaler

# Register Kannada Font
LabelBase.register(name="KannadaFont", fn_regular="NotoSansKannada-Regular.ttf")

# Speech Queue to Prevent Overlapping Speech
speech_queue = queue.Queue()

def speak_kannada(text):
    """Converts text to Kannada speech and plays it asynchronously without overlap."""
    # Ensure text is a string
    text = str(text)
    if text and text.strip() and text != "None":
        print(f"🗣 Speaking: {text}")

        def play_audio():
            try:
                pygame.mixer.init()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                    audio_path = temp_audio.name
                tts = gTTS(text=text, lang='kn')
                tts.save(audio_path)
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.quit()
                time.sleep(0.5)  # Ensure file is not deleted while playing
                os.remove(audio_path)
            except Exception as e:
                print(f"⚠️ Voice Output Error: {e}")

        # Add to queue only if not already busy
        if speech_queue.qsize() == 0:
            speech_queue.put(play_audio)

def process_speech_queue(dt):
    """Process queued speech commands safely."""
    if not speech_queue.empty():
        task = speech_queue.get()
        threading.Thread(target=task, daemon=True).start()

Clock.schedule_interval(process_speech_queue, 1.0)

class DisplayScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # To store last spoken sign to avoid repeating speech
        self.last_spoken_sign = None

        layout = BoxLayout(orientation='vertical', spacing=15, padding=25)

        # Kannada Sign Label – this displays only the Kannada sign prediction.
        self.kannada_label = Label(
            text="Kannada Sign: None",
            font_size=50,
            size_hint=(1, 0.2),
            color=(0, 1, 0, 1),
            font_name="KannadaFont"
        )
        layout.add_widget(self.kannada_label)

        # Buttons
        self.start_button = Button(text="Start Sign Recognition", size_hint=(1, 0.1))
        self.start_button.bind(on_press=self.start_recognition)
        layout.add_widget(self.start_button)

        self.stop_button = Button(text="Stop & Finalize Word", size_hint=(1, 0.1), disabled=True)
        self.stop_button.bind(on_press=self.stop_recognition)
        layout.add_widget(self.stop_button)

        # Camera Feed
        self.img = Image(size_hint=(1, 1))
        layout.add_widget(self.img)

        # Back Button
        self.back_button = Button(
            text="Back to Home",
            size_hint=(1, 0.1),
            background_color=(0.2, 0.6, 1, 1)
        )
        self.back_button.bind(on_press=self.go_back_to_second)
        layout.add_widget(self.back_button)

        self.add_widget(layout)

        # Load Model & Scaler
        BASE_PATH = r"C:\Users\Kingshuk Maji\Documents\Sign_Connect\Sign Connect"
        MODEL_PATH = os.path.join(BASE_PATH, "Models", "SVM", "svm_model.pkl")
        SCALER_PATH = os.path.join(BASE_PATH, "Models", "SVM", "scaler.pkl")

        try:
            self.model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
            if self.model:
                print("✅ Model loaded successfully!")
            else:
                print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None

        try:
            self.scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else StandardScaler()
        except Exception as e:
            print(f"⚠️ Error loading scaler: {e}")
            self.scaler = StandardScaler()

        self.recognition = None
        self.video_capture = None

    def start_recognition(self, instance):
        """Start real-time recognition."""
        if not self.model:
            print("❌ Error: Model is not loaded. Cannot start recognition.")
            return

        self.stop_button.disabled = False
        self.start_button.disabled = True
        self.recognition = RealTimeRecognition(self.model, self.scaler)

        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            print("❌ Error: Webcam not accessible.")
            self.start_button.disabled = False
            self.stop_button.disabled = True
            return

        Clock.schedule_interval(self.update_recognition, 1.0 / 30.0)

    def stop_recognition(self, instance):
        """Stop recognition and release the camera."""
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
        self.stop_button.disabled = True
        self.start_button.disabled = False
        Clock.unschedule(self.update_recognition)

    def update_recognition(self, dt):
        """Update the UI with the Kannada sign and display the webcam feed."""
        if not self.video_capture:
            return

        ret, frame = self.video_capture.read()
        if not ret:
            print("❌ Error: Unable to read frame.")
            return

        # Fix camera orientation (adjust as necessary; here we flip vertically)
        frame = cv2.flip(frame, -1)
        _, kannada_sign = self.recognition.process_frame(frame)

        # Ensure kannada_sign is a string
        kannada_sign = str(kannada_sign)

        # Update label with Kannada sign
        if kannada_sign == "None" or not kannada_sign.strip():
            self.kannada_label.text = "Kannada Sign: No Sign Detected"
        else:
            self.kannada_label.text = f"{kannada_sign}"

        # Trigger speech output only if the sign has changed
        if kannada_sign and kannada_sign != "None" and kannada_sign != self.last_spoken_sign:
            speak_kannada(kannada_sign)
            self.last_spoken_sign = kannada_sign

        # Display the webcam feed
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img.texture = texture

    def go_back_to_second(self, instance):
        """Navigate back to the second screen."""
        self.manager.current = "second"

class SignRecognitionApp(App):
    def build(self):
        return DisplayScreen(name="conversion_screen")

if __name__ == "__main__":
    SignRecognitionApp().run()