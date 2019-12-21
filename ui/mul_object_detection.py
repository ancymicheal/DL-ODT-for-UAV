import glob
import os
import tkFileDialog as fileDialog
import tkMessageBox
from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog
import cv2 as cv
import natsort
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util
from ui.pre_data_mul import PreDataMul

title_font = ("Times New Roman", 18, "bold")
text_font = ("Times New Roman", 14)

pad_x = 10
pad_y = 10


class Mul_Object_Detection(Frame):
    def __init__(self, master, controller, **kw):
        Frame.__init__(self, master, **kw)

        self.controller = controller
        self.grid_columnconfigure(0, weight=1)

        self.grid_rowconfigure(10, weight=1)

        self.selected_video = None
        self.selected_image_folder = None
        self.selected_folder = None


        NUM_CLASSES = PreDataMul.class_name(number_class)
        Prin("###############################################################")
        print(NUM_CLASSES)

        Label(
            self,
            text="MULTIPLE OBJECT DETECTION\n",
            font=title_font,
        ).grid(row=0, column=0, sticky='n', padx=pad_x, pady=pad_y)

        Label(
            self,
            text="Load Image\n"
        ).grid(row=1, column=0, sticky='n', padx=pad_x, pady=pad_y)

        self.mul_image_file_entry = Entry(
            self,
            width=70
        )
        self.mul_image_file_entry.grid(row=2, column=0)
        self.mul_image_file_entry.bind("<Button-1>", self.browse_mul_image_file)

        self.nav_panel = Frame(self)
        self.nav_panel.grid(row=10, column=0, sticky="s")
        Button(
            self.nav_panel,
            text='<< Prev',
            width=10,
            command=self.prev_step
        ).pack(side=LEFT, padx=5, pady=3)

    def prev_step(self):
        from ui.training_mul import TrainingMul
        self.controller.show_frame(TrainingMul)


    def browse_mul_image_file(self, _):
        self.selected_image_file = tkFileDialog.askopenfilename(initialdir = "~/Downloads",
                                                                  title = "Select file",
                                                                  filetypes = (("jpeg files","*.jpg"),("all files","*.*"))
                                                                  )
        if self.selected_image_file:
            self.mul_image_file_entry.delete(0, END)
            self.mul_image_file_entry.insert(0, self.selected_image_file)
