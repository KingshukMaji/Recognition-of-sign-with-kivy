from kivy.uix.screenmanager import Screen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.uix.label import MDLabel
from kivy.graphics import Color, Rectangle

class SecondPageScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Main layout with spacing and padding for better structure
        layout = MDBoxLayout(orientation="vertical", spacing=20, padding=40)
        self.add_background(layout)

        # Title label with bold, larger font
        title_label = MDLabel(
            text="Sign Language Application",
            font_style="H4",
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1),  # White text
            halign="center",
            size_hint=(1, 0.2)
        )
        layout.add_widget(title_label)

        # Button styles
        def create_button(text, callback):
            return MDRectangleFlatButton(
                text=text,
                size_hint=(1, 0.15),
                theme_text_color="Custom",
                text_color=(1, 1, 1, 1),  # White text
                md_bg_color=(0.2, 0.6, 1, 1),  # Blue shade
                on_release=callback
            )

        # Communication button
        conversion_button = create_button("Communication", self.go_to_conversion)
        layout.add_widget(conversion_button)

        # Emergency button
        emergency_button = create_button("Emergency", self.go_to_emergency)
        layout.add_widget(emergency_button)

        # Learning button
        learning_button = create_button("Learning", self.go_to_learning)
        layout.add_widget(learning_button)

        # Back button with a different color
        back_button = MDRectangleFlatButton(
            text="⬅ Back to Login",
            size_hint=(1, 0.1),
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1),
            md_bg_color=(1, 0.34, 0.2, 1),  # Red shade
            on_release=self.go_back_to_auth  
        )

        layout.add_widget(back_button)

        # Add layout to the screen
        self.add_widget(layout)

    def add_background(self, layout):
        """Add dark theme background to layout."""
        with layout.canvas.before:
            Color(0.12, 0.12, 0.12, 1)  # Dark mode background
            self.rect = Rectangle(size=self.size, pos=self.pos)
        layout.bind(size=self.update_rect, pos=self.update_rect)

    def update_rect(self, instance, value):
        """Ensure background resizes correctly."""
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def go_to_conversion(self, instance):
        """Navigate to the communication/conversion screen."""
        self.manager.current = "conversion_screen"

    def go_to_emergency(self, instance):
        """Navigate to the emergency screen."""
        self.manager.current = "emergency_screen"

    def go_to_learning(self, instance):
        """Navigate to the learning/training screen."""
        self.manager.current = "learning_screen"

    def go_back_to_auth(self, instance):
        """Navigate back to the authentication screen."""
        if "auth" in self.manager.screen_names:  
            self.manager.current = "auth"
        else:
            print("❌ Error: 'auth' screen not found in ScreenManager.")

