import tkMessageBox
from Tkinter import *
import Tkinter as tk
import os
import tkFileDialog as filedialog
import cv2
from PIL import Image, ImageTk
import PIL

LARGE_FONT = ("Verdana", 12)
SIZE = (192, 162)

class UploadVideoPage(tk.Frame):
    def __init__(self, master, controller):
        Frame.__init__(self, master)

        self.controller = controller

        self.file_name = "/home/ashok/Data/videos/boat5.avi"
        self.vid = None

        # Title
        self.pageTitle = Label(self, text="Upload Video Page", font=LARGE_FONT)
        self.pageTitle.grid(row=0, column=1, columnspan=2)

        self.input_file = Entry(self)
        self.input_file.grid(row=1, column=0, sticky="we")

        self.browse_button = Button(self, text="Select a video file", command=self.file_browser)
        self.browse_button.grid(row=1, column=1, sticky="we")

        # self.canvas = Canvas(self)
        # self.canvas.grid(row=3, column=0, columnspan=2)

        self.convert_button = Button(self, text="Convert To Frames", command=self.convert_to_frames)
        self.convert_button.grid(row=2, column=0, columnspan=2)

    def convert_to_frames(self):
        vidObj = cv2.VideoCapture(self.file_name)
        count = 0
        success = 1

        base_name = os.path.basename(self.file_name)

        filename = os.path.splitext(base_name)[0]

        self.frame_dir = os.path.dirname(self.file_name) + "/" + filename


        # create directory
        if not os.path.exists(self.frame_dir):
            self.img_folder = os.makedirs(self.frame_dir+'/img')

        while success:
            # function extracts frame
            success, image = vidObj.read()
            if success:
                scale_percent = 60  # percent of original size
                width = int(256)
                height = int(256)
                dim = (width, height)
                resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(self.frame_dir + "/000%d.jpg" % count, resized)

            count += 1

      #  tkMessageBox.showinfo("Success", "Converted to frames successfully.\n Saved in " + self.img_folder)

        self.show_n_frames()

        self.controller.set_frame_directory(self.frame_dir)

    def file_browser(self):
        self.file_name = filedialog.askopenfilename(initialdir="/home",
                                                    title="Select file",
                                                    filetypes=(("avi files", "*.avi"), ("all files", "*.*")))
        self.input_file.delete(0, END)
        self.input_file.insert(0, self.file_name)

    def show_n_frames(self, num_of_frames=6):

        self.canvases = {}
        self.images = {}

        for i in range(0, num_of_frames):
            canvas_key = "canvas"+str(i)
            self.canvases[canvas_key] = Canvas(self, cursor='tcross', width=100, height=100)
            self.canvases[canvas_key].grid(row=3, column=i)

            img_key = "img" + str(i)
            img_path = self.frame_dir + "/000%d.jpg" % i

            img = Image.open(img_path)
            img = img.resize((100, 100), Image.ANTIALIAS)
            self.images[img_key] = ImageTk.PhotoImage(img)
            # self.canvases[canvas_key].config(width=max(self.images[img_key].width(), 400), height=max(self.images[img_key].height(), 400))
            self.canvases[canvas_key].create_image(0, 0, anchor=NW, image=self.images[img_key])

if __name__ == '__main__':
    root = Tk()

    # window properties
    root.geometry('1280x800')
    root.resizable(width=True, height=True)

    tool = UploadVideoPage(root)
    tool.grid(row=0, column=0, sticky="nsew")

    tool.frame_dir = "/home/ashok/Data/videos/boat5/001"
    tool.show_n_frames()

    root.mainloop()
