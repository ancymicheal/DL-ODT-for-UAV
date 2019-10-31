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

from object_detection.utils import label_map_util

from models.research.object_detection.generate_tfrecord import convert

title_font = ("Times New Roman", 18, "bold")
text_font = ("Times New Roman", 14)

pad_x = 10
pad_y = 10
label_file = "/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research/object_detection/labelmap.pbtxt"

frames_train = "/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research/object_detection/images/train/"
frames_test = "/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research/object_detection/images/test/"
image_path = "/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research/object_detection/images/"



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
            label_file_txt.write("item {\n  id: %d\n  name: \'%s\'\n}\n\n" % (i, classes[j].strip()))
            i = i + 1
        label_file_txt.close()

        self.generate_train_test_tf_records()

        self.update_config_file()

        tkMessageBox.showinfo(
                "Success",
                "Classes, TF records and Config file created successfully"
            )

    def generate_train_test_tf_records(self):
        test_csv_path = "/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research/object_detection/images/test_labels.csv"
        test_image_path = "/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research/object_detection/images/test/"
        test_output_path = "/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research/object_detection/test.record"
        train_csv_path = "/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research/object_detection/images/train_labels.csv"
        train_image_path = "/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research/object_detection/images/train/"
        train_output_path = "/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research/object_detection/train.record"
        convert(test_csv_path, test_image_path, test_output_path)
        convert(train_csv_path, train_image_path, train_output_path)

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
        tkMessageBox.showinfo("INFO", "Work in progress")

    def update_config_file(self):
        # Number of classes
        CWD_PATH = "/home/ancy/PycharmProjects/DL-ODT-for-UAV"
        PATH_TO_LABELS = os.path.join(CWD_PATH, 'models', 'research', 'object_detection', 'labelmap.pbtxt')
        label_map = label_map_util.get_label_map_dict(PATH_TO_LABELS)

        # Number of examples
        test_dir = "/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research/object_detection/images/test"

        config_file = "/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research/object_detection/training/faster_rcnn_inception_v2_pets.config"

        config = open(config_file, "r")
        lines = []
        for line in config:

            if "num_classes" in line:
                line = "    num_classes: {}\n".format(len(label_map))

            if "num_examples" in line:
                line = "  num_examples: {}\n".format(len(glob.glob1(test_dir, "*.jpg")))

            lines.append(line)
        config.close()

        write_config = open(config_file, "w")
        for line in lines:
            write_config.write(line)
        write_config.close()

        print("Updated the config file")


