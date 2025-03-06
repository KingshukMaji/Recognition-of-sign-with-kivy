from kivy.uix.boxlayout import BoxLayout
from kivy.uix.video import Video
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
import os

class EmergencyScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', spacing=10, padding=20)

        # ✅ Ensure emergency GIF directory exists
        emergency_gif_dir = r"C:\Users\Kingshuk Maji\Documents\Sign_Connect\Sign Connect\Emergency"

        if not os.path.exists(emergency_gif_dir):
            print(f"❌ Error: Directory '{emergency_gif_dir}' not found.")
            error_label = Label(text="Emergency GIF directory not found!", size_hint=(1, 0.6))
            layout.add_widget(error_label)
        else:
            # ✅ Load GIFs dynamically from the directory
            gif_paths = [os.path.join(emergency_gif_dir, f) for f in os.listdir(emergency_gif_dir) if f.endswith('.gif')]

            if not gif_paths:
                error_label = Label(text="No GIFs found in the directory.", size_hint=(1, 0.6))
                layout.add_widget(error_label)
            else:
                for gif_path in gif_paths:
                    gif_box = BoxLayout(orientation='vertical', size_hint=(1, 0.6))
                    gif = Video(source=gif_path, options={'eos': 'loop'}, size_hint=(1, 0.8))
                    gif.state = 'play'  # ✅ Ensure GIF plays automatically
                    gif_label = Label(text=os.path.splitext(os.path.basename(gif_path))[0], size_hint=(1, 0.2), halign='center')
                    gif_box.add_widget(gif)
                    gif_box.add_widget(gif_label)
                    layout.add_widget(gif_box)

        # ✅ Add Back Button
        back_button = Button(text="⬅ Back to Home", size_hint=(1, 0.1), background_color=(1, 0.3, 0.3, 1))
        back_button.bind(on_press=self.back_to_second_page)
        layout.add_widget(back_button)

        self.add_widget(layout)

    def back_to_second_page(self, instance):
        """Navigate back to the second page safely."""
        if "second" in self.manager.screen_names:
            Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'second'), 0)
        else:
            print("❌ Error: 'second' screen not found in ScreenManager.")
