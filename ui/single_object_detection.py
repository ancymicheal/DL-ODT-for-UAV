from Tkinter import *

from learning.yolo import YoloTf

title_font = ("Times New Roman", 18, "bold")
pad_x = 10
pad_y = 10

canvas_width = 500
canvas_height = 500


class SingleDetection(Frame):
    def __init__(self, master, controller, **kw):
        Frame.__init__(self, master, **kw)

        self.controller = controller
        self.grid_columnconfigure(0, weight=1)

        self.tk_images = []
        self.iterations = 0

        Label(
            self,
            text="YOLO based Single Object Detection",
            font=title_font
        ).grid(row=0, column=0)

        Button(
            self,
            text="Prepare detection training data",
            command=self.prepare_training
        ).grid(row=1, column=0, padx=pad_x, pady=pad_y)

        self.main_panel = Canvas(
            self,
            cursor='tcross',
            highlightbackground="black",
            highlightcolor="black",
            highlightthickness=1,
            width=canvas_width,
            height=canvas_height
        )
        self.main_panel.grid(row=2, column=0)

        # Step Navigation
        self.nav_panel = Frame(self)
        self.nav_panel.grid(row=3, column=0)

        Button(
            self.nav_panel,
            text='<< Prev',
            width=10,
            command=self.prev_step
        ).pack(side=LEFT, padx=5, pady=3)

        Button(
            self.nav_panel,
            text='Next >>',
            width=10,
            command=self.next_step
        ).pack(side=LEFT, padx=5, pady=3)

    def prepare_training(self):
        self.iterations = 0
        YoloTf().prepare_training_data(self)

    def show_location(self):
        image = self.tk_images[self.iterations]
        self.main_panel.config(
            width=max(image.width(), 256), height=max(image.height(), 256)
        )
        self.main_panel.create_image(0, 0, image=image, anchor=NW)
        self.iterations += 1

    def prev_step(self):
        from ui.annotate import Annotate
        self.controller.show_frame(Annotate)

    def next_step(self):
        from ui.single_object_tracking import SingleTracking
        self.controller.show_frame(SingleTracking)
