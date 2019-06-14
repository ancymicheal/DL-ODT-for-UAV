
from kivy.config import Config
import kivy 
from kivy.app import App 
from kivy.uix.label import Label 
from kivy.uix.textinput import TextInput

Config.set('graphics', 'resizable', '1')

Config.set('graphics', 'height', '400')


# defining the App class 
class MyLabelApp(App): 
    def build(self): 
        # label disply the text on screen  
        lbl = Label(text ="[b]Object Detection and Tracking with UAV videos[/b]", markup=True, font_size='30sp') 
	
        return lbl 
  
# creating the object 
label = MyLabelApp() 
# run the window 
label.run() 


