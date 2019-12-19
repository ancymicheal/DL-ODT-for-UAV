import glob
import os, sys
import tkFileDialog as fileDialog
import tkMessageBox
from Tkinter import *
from PIL import Image
import cv2
import cv2 as cv
import natsort
import pandas as pd
import xml.etree.ElementTree as ET

title_font = ("Times New Roman", 18, "bold")
text_font = ("Times New Roman", 14)

pad_x = 10
pad_y = 10

frames_train = "./models/research/object_detection/images/train/"
frames_test = "./models/research/object_detection/images/test/"
image_path = "./models/research/object_detection/images/"



class AnnotateMul(Frame):
    def __init__(self, master, controller, **kw):
        Frame.__init__(self, master, **kw)

        self.controller = controller
        self.grid_columnconfigure(0, weight=1)

        self.grid_rowconfigure(10, weight=1)

        self.selected_video = None
        self.selected_image_folder = None
        self.selected_folder = None

        Label(
            self,
            text="Annotate Dataset\n",
            font=title_font,
        ).grid(row=0, column=0, sticky='n', padx=pad_x, pady=pad_y)

        self.nav_panel = Frame(self)
        self.nav_panel.grid(row=10, column=0, sticky="s")



	# Label the train and test folder 
        Label(
            self,
            text="Label the training and testing images",
            font=text_font
        ).grid(row=1, column=0, sticky='n', padx=pad_x, pady=pad_y)

        Label(
            self,
                text="NOTE: ",
            font=text_font
        ).grid(row=2, column=0, sticky='n', padx=pad_x, pady=pad_y)

        Label(
            self,
            text="Train Image Path : ./models/research/object_detection/images/train/ ",
            font=text_font
        ).grid(row=3, column=0, sticky='n', padx=pad_x, pady=pad_y)

        Label(
            self,
            text="Test Image Path : ./models/research/object_detection/images/test/ ",
            font=text_font
        ).grid(row=4, column=0, sticky='n', padx=pad_x, pady=pad_y)

        self.load_label_button = Button(
            self,
            text="Labeling the train and test images " ,
            command=self.label_mul
        )
        self.load_label_button.grid(row=5, column=0, padx=pad_x, pady=pad_y)

        Button(
            self.nav_panel,
            text='<< Prev',
            width=10,
            command=self.prev_step
        ).pack(side=LEFT, padx=pad_x, pady=pad_y)

        Button(
            self.nav_panel,
            text='Next >>',
            width=10,
            command=self.next_step
        ).pack(side=LEFT, padx=pad_x, pady=pad_y)

    def label_mul(self):
        os.system('./labelimg/labelImg')
   
    def prev_step(self):
        from ui.upload_mul import UploadMul
        self.controller.show_frame(UploadMul)

    def next_step(self):
        from ui.pre_data_mul import PreDataMul
        self.controller.show_frame(PreDataMul)

