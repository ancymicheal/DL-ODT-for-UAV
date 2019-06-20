from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.config import Config
from kivy.uix.videoplayer import VideoPlayer
import os
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
	Button:
		on_release: app.root.current = 'third'
		text: 'Single Object detection and tracking'
		font_size: 20
		pos: 420,390
		
	        
	Label:
	    text: "Object Detection and Tracking"
	    font_size: 30
	    pos_hint: {"x": 0, 'y': .4}

<LoadDialog>:
    BoxLayout:
        size: root.size
        pos: root.pos
        orientation: "vertical"
        FileChooserIconView:
            id: filechooser

        BoxLayout:
            size_hint_y: None
            
            Button:
                text: "Cancel"
                on_release: root.cancel()

            Button:
                text: "Load Video File"
                on_release: root.load(filechooser.path, filechooser.selection)
        BoxLayout:
            TextInput:
                id: text_input
                text: ''

            RstDocument:
                text: text_input.text
                show_errors: True
<root>:
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
	
	Button:
		text: 'Load'
		pos:400,450
		on_release: root.show_load()

	Button:
		text: 'Play'
		pos: 300,450
		on_press: root.video()
  


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

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    text_input = ObjectProperty(None)

class root(Screen):#thridscreen
    loadfile = ObjectProperty(None)
    #savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()


    def load(self, path, filename):
        with open(os.path.join(path, filename[0])) as stream:
            self.text_input.text = stream.read()

        self.dismiss_popup()

    def video(self):
	player = VideoPlayer(source='/home/ancymicheal/boat5.avi',  state='play', options={'allow_stretch': True})
	self._popup = Popup(title="Load file", content=player,
                            size_hint=(0.9, 0.9),auto_dismiss=True )

	self._popup.open()

    	
	
class FourthScreen(Screen):
    pass
class FifthScreen(Screen):
    pass

# Create the screen manager
sm = ScreenManager()
sm.add_widget(MainScreen(name='first'))
sm.add_widget(SecondScreen(name='second'))
sm.add_widget(root(name='third'))#ThirdScreen
sm.add_widget(FourthScreen(name='fourth'))
sm.add_widget(FifthScreen(name='fifth'))

class TestApp(App):

    def build(self):
        return sm

if __name__ == '__main__':
    TestApp().run()
