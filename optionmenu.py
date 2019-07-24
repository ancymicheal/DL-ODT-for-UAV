
from Tkinter import *

master = Tk()

var = StringVar(master)
var.set("one") # initial value

option = OptionMenu(master, var, "one", "two", "three", "four")
option.pack()


def ok():
    print "value -", var.get()
    master.quit()

button = Button(master, text="OK", command=ok)
button.pack()

mainloop()