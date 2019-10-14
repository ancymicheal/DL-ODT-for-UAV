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

frame_resize_dim = (int(500), int(500))

'''def get_video_file_name(path):
    base_name = os.path.basename(path)
    return os.path.splitext(base_name)[0]'''



def directory_exists(directory):
    return os.path.exists(directory)


def get_directories(path):
    return os.listdir(path)


class UploadMul(Frame):
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
            text="Load New Dataset\n"
                 "Upload a new video or select a new image folder",
            font=title_font,
        ).grid(row=0, column=0, sticky='n', padx=pad_x, pady=pad_y)

        self.nav_panel = Frame(self)
        self.nav_panel.grid(row=10, column=0, sticky="s")

	# Upload Video and convert to frames
        Label(
            self,
            text="Video will be converted to frames and copied to the data folder as 80% as train data and 20 % as test data",
            font=text_font,
        ).grid(row=1, column=0, sticky='n', padx=pad_x, pady=pad_y)

        self.video_file_entry = Entry(
            self,
            width=70
        )
        self.video_file_entry.grid(row=2, column=0)
        self.video_file_entry.bind("<Button-1>", self.browse_video_file)

        self.upload_video_button = Button(
            self,
            text="Convert To Frames And Load",
            command=self.convert_to_frames,
        )
        self.upload_video_button.grid(row=3, column=0, padx=pad_x, pady=pad_y)




	# Load a Image Folder
        Label(
            self,
            text="Images will be copied to the data folder as 80% as train data and 20 % as test data",
            font=text_font
        ).grid(row=4, column=0, sticky='n', padx=pad_x, pady=pad_y)

        self.image_folder_entry = Entry(
            self,
            width=70
        )
        self.image_folder_entry.grid(row=5, column=0)
        self.image_folder_entry.bind("<Button-1>", self.browse_image_folder)

        self.load_image_button = Button(
            self,
            text="Load Images",
            command=self.load_images
        )
        self.load_image_button.grid(row=6, column=0, padx=pad_x, pady=pad_y)

	

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

    def browse_video_file(self, _):
        self.selected_video = fileDialog.askopenfilename(
            initialdir="~/Downloads",
            title="Select a video file",
            filetypes=(("avi files", "*.avi"), ("all files", "*.*"))
        )

        if self.selected_video:
            self.video_file_entry.delete(0, END)
            self.video_file_entry.insert(0, self.selected_video)

    def browse_image_folder(self, _):
        self.selected_image_folder = fileDialog.askdirectory(
            title="Select the image folder",
            initialdir="~/Downloads"
        )
        if self.selected_image_folder:
            self.image_folder_entry.delete(0, END)
            self.image_folder_entry.insert(0, self.selected_image_folder)
    def convert_to_frames(self):

        video_object = cv.VideoCapture(self.selected_video)
        counter = 1
        success = 1
	percentage_test = 20
	index_test = round(100 / percentage_test)
	file_num = 1
        
	while success:
		success, image = video_object.read()
		
		if success:
			re_sized = cv.resize(image, frame_resize_dim, interpolation=cv.INTER_AREA)
			if counter == index_test:
				counter = 1
				cv.imwrite(frames_test + "/000%d.jpg" % file_num, re_sized)
			else:
				cv.imwrite(frames_train + "/000%d.jpg" % file_num, re_sized)
				counter = counter + 1
		file_num = file_num+1
	tkMessageBox.showinfo("Success", "Images loaded successfully to /models/research/object_detection/train and /models/research/object_detection/test ")			
	
	
        
            
    def load_images(self):
        path = self.selected_image_folder+"/"
	dirs = os.listdir( path )
	percentage_test = 20
	counter = 1
	index_test = round(100 / percentage_test)
	for item in dirs:
		
        	if os.path.isfile(path+item):
            		im = Image.open(path+item)
			f, e = os.path.splitext(path+item)
	    		item1 = item[:-4]
	    		train_path = frames_train + item1
	    		test_path = frames_test + item1
			
	    		imResize = im.resize((1024,680), Image.ANTIALIAS)
			
	    	
	    		if counter == index_test:
				counter = 1
            			imResize.save(test_path + ' resized.jpg', 'JPEG', quality=90)
				
	    		else:
	        		imResize.save(train_path + ' resized.jpg', 'JPEG', quality=90)
				counter = counter + 1
	
        		
                
        tkMessageBox.showinfo("Success", "Images loaded successfully to /models/research/object_detection/train and /models/research/object_detection/test ")
        #self.controller.working_folder = folder_name
        #from ui.annotate import Annotate
        #self.controller.show_frame(Annotate)

    '''def xml_to_csv(self):
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
	'''
   
    def prev_step(self):
        from ui.select_type import DTType
        self.controller.show_frame(DTType)

    def next_step(self):
        from ui.annotate_mul import AnnotateMul
        self.controller.show_frame(AnnotateMul)

