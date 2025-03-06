from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from auth import login_user

class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Create username and password fields
        self.username_label = Label(text="Username:")
        self.username_input = TextInput(multiline=False)

        self.password_label = Label(text="Password:")
        self.password_input = TextInput(password=True, multiline=False)

        # Create login button
        self.login_button = Button(text="Login")
        self.login_button.bind(on_press=self.login)

        # Create sign-up button (to navigate to the sign-up screen)
        self.signup_button = Button(text="Don't have an account? Sign Up")
        self.signup_button.bind(on_press=self.go_to_signup_screen)

        # Add widgets to layout
        self.layout.add_widget(self.username_label)
        self.layout.add_widget(self.username_input)
        self.layout.add_widget(self.password_label)
        self.layout.add_widget(self.password_input)
        self.layout.add_widget(self.login_button)
        self.layout.add_widget(self.signup_button)

        self.add_widget(self.layout)

    def login(self, instance):
        username = self.username_input.text.strip()
        password = self.password_input.text.strip()

        if login_user(username, password):
            self.manager.current = 'second_page'  # Navigate to the home page after successful login
        else:
            popup = Popup(title="Login Failed",
                          content=Label(text="Invalid username or password!"),
                          size_hint=(0.8, 0.4))
            popup.open()

    def go_to_signup_screen(self, instance):
        """Navigate to the sign-up screen."""
        self.manager.current = 'signup_screen'
