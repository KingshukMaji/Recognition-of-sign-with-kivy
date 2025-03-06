from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.checkbox import CheckBox
from auth import signup_user

class SignUpScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ✅ Main Layout
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=15)

        # ✅ Username Input
        self.username_label = Label(text="Username:", font_size=18, color=(1, 1, 1, 1))
        self.username_input = TextInput(multiline=False, font_size=16)

        # ✅ Password Input
        self.password_label = Label(text="Password:", font_size=18, color=(1, 1, 1, 1))
        self.password_input = TextInput(password=True, multiline=False, font_size=16)

        # ✅ Password Visibility Toggle
        password_toggle_layout = BoxLayout(orientation="horizontal", spacing=10)
        self.show_password_checkbox = CheckBox()
        self.show_password_checkbox.bind(active=self.toggle_password_visibility)
        password_toggle_layout.add_widget(self.show_password_checkbox)
        password_toggle_layout.add_widget(Label(text="Show Password", font_size=16, color=(1, 1, 1, 1)))

        # ✅ Signup Button
        self.signup_button = Button(text="Sign Up", font_size=18, background_color=(0.2, 0.6, 1, 1))
        self.signup_button.bind(on_press=self.signup)

        # ✅ Login Navigation Button
        self.login_button = Button(text="Already have an account? Login", font_size=16, background_color=(1, 0.4, 0.4, 1))
        self.login_button.bind(on_press=self.go_to_login_screen)

        # ✅ Add Widgets to Layout
        self.layout.add_widget(self.username_label)
        self.layout.add_widget(self.username_input)
        self.layout.add_widget(self.password_label)
        self.layout.add_widget(self.password_input)
        self.layout.add_widget(password_toggle_layout)  # Password Toggle
        self.layout.add_widget(self.signup_button)
        self.layout.add_widget(self.login_button)

        self.add_widget(self.layout)

    def toggle_password_visibility(self, checkbox, value):
        """✅ Toggle password visibility."""
        self.password_input.password = not value

    def signup(self, instance):
        """✅ Handle user signup."""
        username = self.username_input.text.strip()
        password = self.password_input.text.strip()

        if not username or not password:
            self.show_popup("Signup Failed", "Username and password cannot be empty!")
            return

        if signup_user(username, password):
            self.show_popup("Signup Successful", "Account created successfully!", success=True)
            self.manager.current = 'login_screen'  # ✅ Navigate to login after signup
        else:
            self.show_popup("Signup Failed", "Username already exists!")

    def show_popup(self, title, message, success=False):
        """✅ Display a popup message."""
        popup = Popup(
            title=title,
            content=Label(text=message, font_size=16),
            size_hint=(0.8, 0.4),
        )
        popup.open()

    def go_to_login_screen(self, instance):
        """✅ Navigate to the login screen."""
        self.manager.current = 'login_screen'