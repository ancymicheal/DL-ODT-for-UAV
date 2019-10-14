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
label_file = "./models/research/object_detection/labelmap.pbtxt"
frames_train = "./models/research/object_detection/images/train/"
frames_test = "./models/research/object_detection/images/test/"
image_path = "./models/research/object_detection/images/"



class TrainingDataMul(Frame):
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
            text="Training Dataset\n",
            font=title_font,
        ).grid(row=0, column=0, sticky='n', padx=pad_x, pady=pad_y)

        self.nav_panel = Frame(self)
        self.nav_panel.grid(row=10, column=0, sticky="s")


	# Convert the .xml file of training and testing data into train.csv and test .csv
        Label(
            self,
            text="Convert the .xml file of training and testing data into train.csv and test .csv",
            font=text_font
        ).grid(row=1, column=0, sticky='n', padx=pad_x, pady=pad_y)

        
        self.load_xmltocsv_button = Button(
            self,
            text="convert .xml to .csv ",
	    command=self.xml_to_csv         
        )
        self.load_xmltocsv_button.grid(row=2, column=0, padx=pad_x, pady=pad_y)

	#Enter the Class Name
	Label(
            self,
            text="Enter the Class name",
            font=text_font
        ).grid(row=3, column=0, sticky='n', padx=pad_x, pady=pad_y)
	
	input_text = StringVar()
	self.class_name_entry = Entry(
            self,
            width=10,
	    text = input_text)
	
	self.class_name_entry.grid(row=4, column=0)
        self.class_name_entry.bind("<Button-1>")
	self.class_name_button = Button(
            self,
            text="OK",
            command=self.class_name
        )
        self.class_name_button.grid(row=5, column=0, padx=pad_x, pady=pad_y)
	
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

    def class_name(self):
	entry = self.class_name_entry.get()
	classes = entry.split(",")

	label_file_txt = open(label_file,"w")

	i = 1
	for j in range(0, len(classes)):
		label_file_txt.write("item {\n\tid: %d\n\tname: \'%s\'\n}\n" % (i, classes[j].strip()))
		i = i + 1
	label_file_txt.close()
	tkMessageBox.showinfo(
                "Success",
                "Classes saved successfully"
            )

    def xml_to_csv(self):
	for folder in ['train','test']:
		image_path_folder = os.path.join(image_path + folder)
	
		xml_list = []
    		for xml_file in glob.glob(image_path_folder + '/*.xml'):
        		tree = ET.parse(xml_file)
        		root = tree.getroot()
        		for member in root.findall('object'):
            			value = (root.find('filename').text,
                     			int(root.find('size')[0].text),
                     			int(root.find('size')[1].text),
                     			member[0].text,
                     			int(member[4][0].text),
                     			int(member[4][1].text),
                     			int(member[4][2].text),
                     			int(member[4][3].text)
                     			)
            			xml_list.append(value)
    		column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    		xml_df = pd.DataFrame(xml_list, columns=column_name)
        	xml_df.to_csv((image_path + folder + '_labels.csv'), index=None)
        tkMessageBox.showinfo("Success","Successfully converted .xml to .csv")
		
   
    def prev_step(self):
        from ui.annotate_mul import AnnotateMul
        self.controller.show_frame(AnnotateMul)

    def next_step(self):
        tkMessageBox.showinfo("INFO", "Work on progress")


