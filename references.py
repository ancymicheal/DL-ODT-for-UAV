from Tkinter import *
import Tkinter as tk
import os

LARGE_FONT = ("Verdana", 12)


class References(tk.Frame):
    def __init__(self, master, controller):
        Frame.__init__(self, master)

        self.controller = controller

        # Title
        self.ref_pageTitle = Label(self, text="REFERENCES", font=LARGE_FONT)
        self.ref_pageTitle.grid(row=0, column=0)

        self.ref_scrollbar = Scrollbar(self)

        self.configfile = Text(self, wrap=WORD, yscrollcommand=self.ref_scrollbar.set)
        ref_filename = os.getcwd() + '/references.txt'

        with open(ref_filename, 'r') as f:
            self.configfile.insert(INSERT, f.read())
        self.configfile.grid(row=1, column=0)
        self.ref_scrollbar.grid(row=1, column=1)
        self.ref_scrollbar.config(command=self.configfile.yview)

        self.configfile.config(state=DISABLED)
        # Scrollable Text Area

        # Read Text from the file

        # Update the text area


if __name__ == '__main__':
    root = Tk()

    # window properties
    root.geometry('1280x800')
    root.resizable(width=True, height=True)

    tool = References(root)
    tool.grid(row=0, column=0, sticky="nsew")

    root.mainloop()
