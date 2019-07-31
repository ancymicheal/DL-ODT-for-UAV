import Tkinter as tk
import os
from Tkinter import *

title_font = ("Times New Roman", 18, "bold")
text_font = ("Times New Roman", 16)
pad_x = 20
pad_y = 20


class Steps(tk.Frame):
    def __init__(self, master, controller):
        tk.Frame.__init__(self, master, highlightbackground="black",
                          highlightcolor="black",
                          highlightthickness=1)

        self.controller = controller

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Title
        self.page_title = Label(self, text="STEPS", font=title_font)
        self.page_title.grid(row=0, column=0, sticky='nsew', padx=pad_x, pady=pad_y)

        # Scrollable Text Area
        self.scrollbar = Scrollbar(self)
        self.steps_text = Text(self, wrap=WORD, yscrollcommand=self.scrollbar.set)

        filename = os.getcwd() + '/steps.txt'
        with open(filename, 'r') as f:
            self.steps_text.insert(INSERT, f.read())

        self.steps_text.grid(row=1, column=0, sticky="nsew")
        self.steps_text.config(state=DISABLED, font=text_font)

        self.scrollbar.grid(row=1, column=1, sticky="nsew")
        self.scrollbar.config(command=self.steps_text.yview)

        # Step Navigation
        self.nav_panel = Frame(self)
        self.nav_panel.grid(row=2, column=0)

        Button(
            self.nav_panel,
            text='Next >>',
            width=10,
            command=self.next_step
        ).pack(side=LEFT, padx=pad_x, pady=pad_y)

    def next_step(self):
        from ui.select_type import DTType
        self.controller.show_frame(DTType)
