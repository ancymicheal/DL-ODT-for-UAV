import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.config import Config

Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'width', '900')

# Create both screens. Please note the root.manager.current: this is how
# you can control the ScreenManager from kv. Each screen has by default a
# property manager that gives you the instance of the ScreenManager used.
Builder.load_string("""
<Button>:
    font_size: 30
	color: 0,1,0,1
	size_hint: 0.08, 0.1 
<MainScreen>:
    name: 'first'
	Button:
		on_release: app.root.current = 'second'
		text: 'Next'
		pos: 800, 0
	Label:
	    text: "Title"
	    font_size: 30
	    pos_hint: {"x": 0, 'y': .4}

<SecondScreen>:
    name: 'second'
	Button:
		on_release: app.root.current = 'third'
		text: 'Next'
		pos: 800, 0
	Button:
		on_release: app.root.current = 'first'
		text: 'Prev'
		pos: 30, 0
	Label:
	    text: "Object Detection and Tracking"
	    font_size: 30
	    pos_hint: {"x": 0, 'y': .4}

<ThirdScreen>:
    name: 'third'
	Button:
		on_release: app.root.current = 'fourth'
		text: 'Next'
		pos: 800, 0
	Button:
		on_release: app.root.current = 'second'
		text: 'Prev'
		pos: 30, 0
	Label:
	    text: "Upload Video"
	    font_size: 30
	    pos_hint: {"x": 0, 'y': .4}
<FourthScreen>:
    name: 'fourth'
	Button:
		on_release: app.root.current = 'fifth'
		text: 'Next'
		pos: 800, 0
	Button:
		on_release: app.root.current = 'third'
		text: 'Prev'
		pos: 30, 0
	Label:
	    text: "Frame Conversion"
	    font_size: 30
	    pos_hint: {"x": 0, 'y': .4}
<FifthScreen>:
    name: 'fifth'
	Button:
		on_release: app.root.current = 'fifth'
		text: 'Next'
		pos: 800, 0
	Button:
		on_release: app.root.current = 'fourth'
		text: 'Prev'
		pos: 30, 0
	Label:
	    text: "Annotate"
	    font_size: 30
	    pos_hint: {"x": 0, 'y': .4}
""")


# Declare both screens
class MainScreen(Screen):
    pass


class SecondScreen(Screen):
    pass


class ThirdScreen(Screen):
    pass


class FourthScreen(Screen):
    pass


class FifthScreen(Screen):
    pass


# Create the screen manager
sm = ScreenManager()
sm.add_widget(MainScreen(name='first'))
sm.add_widget(SecondScreen(name='second'))
sm.add_widget(ThirdScreen(name='third'))
sm.add_widget(FourthScreen(name='fourth'))
sm.add_widget(FifthScreen(name='fifth'))


class TestApp(App):

    def build(self):
        return sm


if __name__ == '__main__':
    TestApp().run()
