import glob
import os
import tkFileDialog as fileDialog
import tkMessageBox
from Tkinter import *

import cv2
import cv2 as cv
import natsort

title_font = ("Times New Roman", 18, "bold")
text_font = ("Times New Roman", 14)

pad_x = 10
pad_y = 10

data_directory = "./rolo/data"

frame_resize_dim = (int(500), int(500))


def get_video_file_name(path):
    base_name = os.path.basename(path)
    return os.path.splitext(base_name)[0]


def get_directory(filename):
    return data_directory + "/" + filename + "/" + "img"


def directory_exists(directory):
    return os.path.exists(directory)


def get_directories(path):
    return os.listdir(path)


class Upload(Frame):
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

        # Upload Video and convert to frames
        Label(
            self,
            text="Video will be converted to frames and copied to the data folder",
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
            text="Images will be copied to the data folder",
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

        # Select Existing Dataset
        Label(
            self,
            text="Select Existing Dataset",
            font=title_font
        ).grid(row=7, column=0, sticky='n', padx=pad_x, pady=pad_y)

        # drop down
        data_set = get_directories(data_directory)
        self.selected_folder = StringVar(self)
        if len(data_set) > 0:
            self.selected_folder.set(data_set[0])
        OptionMenu(
            self,
            self.selected_folder,
            *data_set
        ).grid(row=8, column=0)
        Button(
            self,
            text="Load Dataset",
            command=self.load_dataset
        ).grid(row=9, column=0, padx=pad_x, pady=pad_y)

        # todo, show images on drop down select

        # Step Navigation
        self.nav_panel = Frame(self)
        self.nav_panel.grid(row=10, column=0, sticky="s")

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

    def prev_step(self):
        from ui.select_type import DTType
        self.controller.show_frame(DTType)

    def next_step(self):
        from ui.annotate import Annotate
        self.controller.show_frame(Annotate)

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
        count = 0
        success = 1

        filename = get_video_file_name(self.selected_video)
        directory = get_directory(filename)
        if directory_exists(directory):
            tkMessageBox.showinfo("ERROR", "Dataset with name " + filename +
                                  " already exists.\n Rename the file and try again")
        else:
            self.controller.show_progress(True)
            os.makedirs(directory)

            # extract and resize frame
            while success:
                success, image = video_object.read()
                if success:
                    re_sized = cv.resize(image, frame_resize_dim, interpolation=cv.INTER_AREA)
                    cv.imwrite(directory + "/000%d.jpg" % count, re_sized)
                    count += 1

            self.controller.show_progress(False)
            tkMessageBox.showinfo(
                "Success",
                "Converted to frames successfully."
            )
            self.controller.working_folder = data_directory + "/" + filename
            from ui.annotate import Annotate
            self.controller.show_frame(Annotate)

    def load_images(self):
        folder_name = data_directory + "/" + os.path.basename(self.selected_image_folder)
        if os.path.exists(folder_name):
            tkMessageBox.showinfo("ERROR", "Folder name already exists. "
                                           "Change the folder name and try again.")
        else:
            os.makedirs(folder_name + "/img")
            files = natsort.natsorted(
                [img for img in glob.glob(self.selected_image_folder + "/*.jpg")],
                reverse=False
            )
            i = 0
            for f in files:
                image = cv2.imread(f)
                resized = cv2.resize(image, frame_resize_dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(folder_name + "/img/000%d.jpg" % i, resized)
                i = i + 1
            tkMessageBox.showinfo("Success", "Images loaded successfully")
            self.controller.working_folder = folder_name
            from ui.annotate import Annotate
            self.controller.show_frame(Annotate)

    def load_dataset(self):
        self.controller.working_folder = data_directory + "/" + self.selected_folder.get()
        from ui.annotate import Annotate
        self.controller.show_frame(Annotate)

