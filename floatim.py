from kivy.app import App
#kivy.require("1.8.0")
from kivy.uix.floatlayout import FloatLayout

class simple(App):
    def build(self):
        return FloatLayout()

if __name__ == "__main__":
    simple().run()
