from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle
from kivy.properties import ListProperty
from kivy.core.window import Window  # ✅ Required for mouse tracking

class Hoverable(ButtonBehavior, Label):
    hover_color = ListProperty([1, 0, 0, 1])  # Default hover color (Red)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._normal_color = self.color  # Store the normal label color
        self._normal_bg_color = [1, 1, 1, 1]  # Default background color (White)
        
        # ✅ Bind window mouse movement to detect hover
        Window.bind(mouse_pos=self.on_mouse_move)
        
        # ✅ Ensure background updates when size/position changes
        self.bind(size=self._update_rect, pos=self._update_rect)

    def on_mouse_move(self, window, pos):
        """Detect mouse position and apply hover effect."""
        if self.collide_point(*pos):  # ✅ Check if mouse is over the widget
            self.on_enter()
        else:
            self.on_leave()

    def on_enter(self):
        """Change color when mouse enters the widget."""
        self._apply_hover_color()

    def on_leave(self):
        """Reset color when mouse leaves the widget."""
        self._apply_normal_color()

    def _apply_hover_color(self):
        """Apply hover background and text color."""
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self.hover_color)  # ✅ Background color
            self.rect = Rectangle(size=self.size, pos=self.pos)
        
        self.color = [0, 0, 0, 1]  # ✅ Change text color to Black for contrast

    def _apply_normal_color(self):
        """Reset to normal colors."""
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self._normal_bg_color)  # ✅ Reset background to white
            self.rect = Rectangle(size=self.size, pos=self.pos)
        
        self.color = self._normal_color  # ✅ Restore text color

    def _update_rect(self, instance, value):
        """Ensure the rectangle follows the label's size and position."""
        if hasattr(self, 'rect'):
            self.rect.pos = self.pos
            self.rect.size = self.size