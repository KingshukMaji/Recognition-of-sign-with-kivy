import json
import os
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivymd.app import MDApp
from Second_page import SecondPageScreen
from Display import DisplayScreen
from emergency import EmergencyScreen
from training import LearningScreen
from kivy.core.window import Window

# Constants
USERS_FILE = "users.json"
Window.set_icon("assets/icon.png")
# Ensure users.json exists
def initialize_users_file():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({"users": []}, f)

initialize_users_file()

class AuthScreen(Screen):
    """Unified Login & Signup Screen with Tabs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation="vertical", padding=20, spacing=10)
        self.tabs = TabbedPanel(do_default_tab=False)

        # Login Tab
        login_tab = TabbedPanelItem(text="Login")
        login_tab_layout = self.create_login_ui()
        login_tab.add_widget(login_tab_layout)

        # Signup Tab
        signup_tab = TabbedPanelItem(text="Sign Up")
        signup_tab_layout = self.create_signup_ui()
        signup_tab.add_widget(signup_tab_layout)

        # Add icons above tabs
        icon_layout = BoxLayout(orientation="horizontal", size_hint_y=None, height=100)
        icon_layout.add_widget(Image(source="login_icon.png", size_hint=(None, None), size=(50, 50)))
        icon_layout.add_widget(Image(source="signup_icon.png", size_hint=(None, None), size=(50, 50)))

        layout.add_widget(icon_layout)
        self.tabs.add_widget(login_tab)
        self.tabs.add_widget(signup_tab)

        layout.add_widget(self.tabs)
        self.add_widget(layout)

    def create_login_ui(self):
        """Login UI with checkbox for password visibility."""
        layout = BoxLayout(orientation="vertical", spacing=10)

        layout.add_widget(Label(text="Username", font_size=18))
        self.login_username = TextInput(hint_text="Enter Username", font_size=16, multiline=False)
        layout.add_widget(self.login_username)

        layout.add_widget(Label(text="Password", font_size=18))
        password_box = BoxLayout(orientation="horizontal", spacing=10)
        self.login_password = TextInput(hint_text="Enter Password", password=True, font_size=16, multiline=False)
        self.password_checkbox = CheckBox()
        self.password_checkbox.bind(active=self.toggle_password_visibility)
        password_box.add_widget(self.login_password)
        password_box.add_widget(self.password_checkbox)
        layout.add_widget(password_box)

        login_button = Button(text="Login", background_color=(0.2, 0.6, 1, 1), size_hint=(1, 0.2))
        login_button.bind(on_press=self.validate_login)
        layout.add_widget(login_button)

        return layout

    def create_signup_ui(self):
        """Signup UI with checkbox for password visibility."""
        layout = BoxLayout(orientation="vertical", spacing=10)

        layout.add_widget(Label(text="Username", font_size=18))
        self.signup_username = TextInput(hint_text="Choose a Username", font_size=16, multiline=False)
        layout.add_widget(self.signup_username)

        layout.add_widget(Label(text="Password", font_size=18))
        password_box = BoxLayout(orientation="horizontal", spacing=10)
        self.signup_password = TextInput(hint_text="Create a Password", password=True, font_size=16, multiline=False)
        self.password_checkbox_signup = CheckBox()
        self.password_checkbox_signup.bind(active=self.toggle_password_visibility_signup)
        password_box.add_widget(self.signup_password)
        password_box.add_widget(self.password_checkbox_signup)
        layout.add_widget(password_box)

        signup_button = Button(text="Sign Up", background_color=(1, 0.34, 0.2, 1), size_hint=(1, 0.2))
        signup_button.bind(on_press=self.create_account)
        layout.add_widget(signup_button)

        return layout

    def toggle_password_visibility(self, instance, value):
        """Toggle password visibility in Login."""
        self.login_password.password = not value

    def toggle_password_visibility_signup(self, instance, value):
        """Toggle password visibility in Signup."""
        self.signup_password.password = not value

    def validate_login(self, instance):
        """Validate login details and navigate to the second page."""
        username, password = self.login_username.text.strip(), self.login_password.text.strip()

        if not username or not password:
            self.show_popup("Login Failed", "Username and password cannot be empty.")
            return

        try:
            with open(USERS_FILE, "r") as f:
                users = json.load(f)["users"]
        except (json.JSONDecodeError, FileNotFoundError):
            self.show_popup("Error", "Error reading user data.")
            return

        for user in users:
            if user["username"] == username and user["password"] == password:
                if "second" in self.manager.screen_names:
                    self.manager.current = "second"
                else:
                    self.show_popup("Error", "Second screen not found!")
                return

        self.show_popup("Login Failed", "Invalid username or password.")

    def create_account(self, instance):
        """Signup process."""
        username, password = self.signup_username.text.strip(), self.signup_password.text.strip()

        if not username or not password:
            self.show_popup("Signup Failed", "Username and password cannot be empty.")
            return

        try:
            with open(USERS_FILE, "r") as f:
                users = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self.show_popup("Error", "Error reading user data.")
            return

        if any(user["username"] == username for user in users["users"]):
            self.show_popup("Signup Failed", "Username already exists.")
            return

        users["users"].append({"username": username, "password": password})

        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)

        self.show_popup("Signup Successful", "Account created successfully!")

    def show_popup(self, title, message):
        """Display popups for errors/messages."""
        popup = Popup(title=title, content=Label(text=message, font_size=18), size_hint=(0.8, 0.4))
        popup.open()

class SignConnectApp(MDApp):  
    def build(self):
        sm = ScreenManager()
        sm.add_widget(AuthScreen(name="auth"))
        sm.add_widget(SecondPageScreen(name="second"))
        sm.add_widget(DisplayScreen(name="conversion_screen"))
        sm.add_widget(EmergencyScreen(name="emergency_screen"))
        sm.add_widget(LearningScreen(name="learning_screen"))
        return sm


if __name__ == "__main__":
    SignConnectApp().run()
