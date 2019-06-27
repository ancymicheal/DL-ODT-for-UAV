from Tkinter import *
import Tkinter as tk
import os

LARGE_FONT = ("Verdana", 12)


class StartPage(tk.Frame):
    def __init__(self, master, controller):
        Frame.__init__(self, master)

        self.controller = controller

        # Title
        self.pageTitle = Label(self, text="STEPS", font=LARGE_FONT)
        self.pageTitle.grid(row=0, column=0)

        self.scrollbar = Scrollbar(self)

        self.configfile = Text(self, wrap=WORD, yscrollcommand=self.scrollbar.set)
        filename = os.getcwd() + '/steps.txt'

        with open(filename, 'r') as f:
            self.configfile.insert(INSERT, f.read())
        self.configfile.grid(row=1, column=0)
        self.scrollbar.grid(row=1, column=1)
        self.scrollbar.config(command=self.configfile.yview)

        # Scrollable Text Area

        # Read Text from the file

        # Update the text area


if __name__ == '__main__':
    root = Tk()

    # window properties
    root.geometry('1280x800')
    root.resizable(width=True, height=True)

    tool = StartPage(root)
    tool.grid(row=0, column=0, sticky="nsew")

    root.mainloop()
