import glob
import tkMessageBox
from Tkinter import *
import Tkinter as tk
import os
import tkFileDialog as filedialog
import cv2
from PIL import Image, ImageTk
import PIL
from natsort import natsort

LARGE_FONT = ("Verdana", 12)
SIZE = (192, 162)


class UploadVideoPage(tk.Frame):
    def __init__(self, master, controller):
        Frame.__init__(self, master)

        self.controller = controller

        self.vid = None
        global variable_annotate_folder
        # Title
        self.pageTitle = Label(self, text="Upload File", font=LARGE_FONT)
        self.pageTitle.grid(row=0, column=1)
        self.pageTitle1 = Label(self, text="Upload Video File")
        self.pageTitle1.grid(row=1, column=0, sticky="nw")
        self.input_file = Entry(self)
        self.input_file.grid(row=1, column=1, sticky="ne")


        self.browse_button = Button(self, text="Select a video file", command=self.file_browser)
        self.browse_button.grid(row=1, column=2, sticky="we")

        # self.canvas = Canvas(self)
        # self.canvas.grid(row=3, column=0, columnspan=2)

        self.convert_button = Button(self, text="Frame Conversion", command=self.convert_to_frames)
        self.convert_button.grid(row=1, column=3)
        self.pageTitle2 = Label(self, text="Upload Image Folder")
        self.pageTitle2.grid(row=2, column=0, sticky="nw")

        self.image_folder_entry = Entry(self)
        self.image_folder_entry.grid(row=2, column=1, sticky="we")

        self.browse_img_button = Button(self, text="Load Image Folder", command=self.load_image_folder)
        self.browse_img_button.grid(row=2, column=2)
        self.pageTitle3 = Label(self, text="Load Annotated Image Folder")
        self.pageTitle3.grid(row=3, column=0, sticky="nw")
        variable_annotate_folder = StringVar(self)
        a = os.listdir("./ROLO/DATA/")

        variable_annotate_folder.set(a[0])
        self.tracking_option = OptionMenu(self, variable_annotate_folder, *a)
        self.tracking_option.grid(row=3, column=1,  sticky="news")
        self.tracking_button = Button(self, text="Ok", command=self.ok)
        self.tracking_button.grid(row=3, column=2, sticky="news")


    def ok(self):

        b = variable_annotate_folder.get()
        folder_path = os.path.dirname("./ROLO/DATA/") + '/' + b + '/'
        img_annotate_path = os.path.join(folder_path) + 'img/'
        labels_path = os.path.join(folder_path) + 'labels/'

        print(img_annotate_path)
        print(labels_path)
        self.controller.set_frame_directory(img_annotate_path)
        #self.controller.set_frame_directory(self.frame_dir)
    def file_browser(self):
        self.file_name = filedialog.askopenfilename(initialdir="/home",
                                                    title="Select file",
                                                    filetypes=(("avi files", "*.avi"), ("all files", "*.*")))

        self.input_file.delete(0, END)
        self.input_file.insert(0, self.file_name)

    def convert_to_frames(self):

        vidObj = cv2.VideoCapture(self.file_name)
        count = 0
        success = 1

        base_name = os.path.basename(self.file_name)

        filename = os.path.splitext(base_name)[0]
        self.frame_dir2 = "./ROLO/DATA/"
        self.frame_dir = os.path.dirname(self.frame_dir2) + "/" + filename + "/" + "img"

        # create directory
        if os.path.exists(self.frame_dir):

            tkMessageBox.showinfo("ERROR", "File already exists")
        else:
            os.makedirs(self.frame_dir)
            while success:
                # function extracts frame
                success, image = vidObj.read()
                if success:
                    width = int(500)
                    height = int(500)
                    dim = (width, height)
                    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

                    cv2.imwrite(self.frame_dir + "/000%d.jpg" % count, resized)

                count += 1

            tkMessageBox.showinfo("Success", "Converted to frames successfully.\n Saved in " + self.frame_dir)

            self.show_n_frames()

        self.controller.set_frame_directory(self.frame_dir)

    def load_image_folder(self):
        self.image_folder = filedialog.askdirectory(title="Select the image folder")
        self.image_folder_entry.delete(0, END)
        self.image_folder_entry.insert(0, self.image_folder)

        # check if the folder name exists in ROLO/data
        folder_name = "./ROLO/DATA/" + os.path.basename(self.image_folder)
        if os.path.exists(folder_name):
            tkMessageBox.showinfo("ERROR", "Folder name already exists. "
                                           "Change the folder name and try again.")
        else:
            # show progress bar

            # resize and copy to ROLO/data/img_folder_name/img
            os.makedirs(folder_name + "/img")
            width = int(500)
            height = int(500)
            dim = (width, height)
            filenames = [img for img in glob.glob(self.image_folder + "/*.jpg")]
            files = natsort.natsorted(filenames, reverse=False)
            i = 0
            for f in files:
                image = cv2.imread(f)
                resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(folder_name + "/img/000%d.jpg" % i, resized)
                i = i + 1

            # load newly created frames directory
            self.controller.set_frame_directory(folder_name + "/img")

            # stop progress bar

            # show success message box
            tkMessageBox.showinfo("Success", "Images loaded successfully")

    def show_n_frames(self, num_of_frames=6):

        self.canvases = {}
        self.images = {}

        for i in range(0, num_of_frames):
            canvas_key = "canvas" + str(i)
            self.canvases[canvas_key] = Canvas(self, cursor='tcross', width=100, height=100)
            self.canvases[canvas_key].grid(row=3, column=i)

            img_key = "img" + str(i)
            img_path = self.frame_dir + "/000%d.jpg" % i

            img = Image.open(img_path)
            img = img.resize((100, 100), Image.ANTIALIAS)
            self.images[img_key] = ImageTk.PhotoImage(img)
            # self.canvases[canvas_key].config(width=max(self.images[img_key].width(), 400),
            # height=max(self.images[img_key].height(), 400)).
            self.canvases[canvas_key].create_image(0, 0, anchor=NW, image=self.images[img_key])


if __name__ == '__main__':
    root = Tk()

    # window properties
    # root.geometry('820x820')
    # root.resizable(width=True, height=True)

    tool = UploadVideoPage(root)
    tool.grid(row=0, column=0, sticky="nsew")

    tool.frame_dir = "/home/ashok/Data/videos/boat5/001"
    tool.show_n_frames()

    root.mainloop()

