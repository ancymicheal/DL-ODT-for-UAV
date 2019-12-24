import glob
import os
import tkFileDialog as fileDialog
import tkMessageBox
from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog
import cv2
import natsort
import numpy as np
import tensorflow as tf
import sys
from PIL import Image, ImageTk
#from models.research.object_detection.Object_detection_image import obj_detect_mul
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util

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
canvas_width = 500
canvas_height = 500

class Mul_Object_Detection(Frame):
    def __init__(self, master, controller, **kw):
        Frame.__init__(self, master, **kw)

        self.controller = controller
        self.grid_columnconfigure(0, weight=1)

        self.grid_rowconfigure(10, weight=1)

        self.selected_video = None
        self.selected_image_folder = None
        self.selected_folder = None
        self.main_panel = Canvas(
            self,
            cursor='tcross',
            highlightbackground="black",
            highlightcolor="black",
            highlightthickness=1,
            width=canvas_width,
            height=canvas_height
        )
        self.main_panel.grid(row=4, column=0)




        #NUM_CLASSES = PreDataMul.class_name(number_class)
        print("###############################################################")
        #print(NUM_CLASSES)

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

        '''Label(
            self,
            text="Before Object Detection\n"
        ).grid(row=3, column=0, sticky='n', padx=pad_x, pady=pad_y)'''

        '''self.training = Button(self, text="DETECT OBJECT", command=self.object_detect_show)
        self.training.grid(row=3, column=0, padx=pad_x, pady=pad_y)'''

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
    #def object_detect_show(self):

        CWD_PATH = os.getcwd()

        #IMAGE_NAME = '000001.jpg'

        PATH_TO_CKPT =  os.path.join(CWD_PATH,'models','research','object_detection','inference_graph','frozen_inference_graph.pb')
        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH, 'models', 'research', 'object_detection', 'training', 'labelmap.pbtxt')

        # Path to image
        PATH_TO_IMAGE = self.selected_image_file

        NUM_CLASSES = 2

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = cv2.imread(PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Draw the results of the detection (aka 'visulaize the results')

        image_detected = vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        #image = Image.open(image)
        '''self.tk_image = ImageTk.PhotoImage(image)
        self.main_panel.config(
            width=max(image.width(), 256), height=max(image.width(), 256))
        self.main_panel.create_image(0, 0, image=self.tk_image, anchor=NW)'''



        cv2.imwrite(os.path.join(CWD_PATH , 'waka.jpg'), image)

        image = Image.open(os.path.join(CWD_PATH , 'waka.jpg'))
        self. tk_image = ImageTk.PhotoImage(image)
        self.main_panel.config(
            width=max(self.tk_image.width(), 256), height=max(self.tk_image.height(), 256)
        )
        self.main_panel.create_image(0, 0, image=self.tk_image, anchor=NW)



        #cv2.waitKey(0)
        #scv2.destroyAllWindows()




'''
    def load_mul_image(self):
            detect_image = obj_detect_mul(image)
            image = Image.open(detect_image)
            resize_value = (500,256)
            #image_resize = image.resize(resize_value)
            self.tk_image = ImageTk.PhotoImage(image)
            #self.main_panel.config(
            #width=max(image.width(), 256), height=max(image.width(), 256))
            self.main_panel.config(
                width=500, height=500)
            self.main_panel.create_image(0, 0, image=self.tk_image, anchor=NW)'''



