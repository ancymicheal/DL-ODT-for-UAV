import glob
import os
import tkMessageBox
from Tkinter import *

import natsort
from PIL import Image, ImageTk

from learning.yolo import YoloTf

title_font = ("Times New Roman", 18, "bold")
text_font = ("Times New Roman", 14)

pad_x = 5
pad_y = 5

data_directory = "./rolo/data"

canvas_width = 448
canvas_height = 448

# colors for the bbox
COLOR = 'red'


class Annotate(Frame):
    def __init__(self, master, controller, **kw):
        Frame.__init__(self, master, **kw)

        self.controller = controller
        self.grid_columnconfigure(0, weight=1)

        # Initialize
        self.yolo = YoloTf()
        self.cur = 0
        self.total = 0
        self.images = []
        self.tk_image = None
        self.label_directory = None
        self.bbox = None
        self.bbox_id = None
        self.label_file = None
        self.mouse = {
            'click': 0, 'x': 0, 'y': 0
        }
        self.hl = None
        self.vl = None

        self.bind("<<ShowFrame>>", self.on_show_frame)

        # Title
        Label(
            self,
            text="Annotate Images",
            font=title_font
        ).grid(row=0, column=0)

        # Main Panel
        self.main_panel = Canvas(
            self,
            cursor='tcross',
            highlightbackground="black",
            highlightcolor="black",
            highlightthickness=1,
            width=canvas_width,
            height=canvas_height
        )
        self.main_panel.bind("<Button-1>", self.mouse_click)
        self.main_panel.bind("<Motion>", self.mouse_move)
        self.bind("<Escape>", self.cancel_bbox)  # press <Espace> to cancel current bbox
        self.bind("s", self.cancel_bbox)
        self.bind("a", self.prev_image)  # press 'a' to go back
        self.bind("d", self.next_image)  # press 'd' to go forward
        self.main_panel.grid(row=1, column=0)

        # Image Navigation
        self.control_panel = Frame(self)
        self.control_panel.grid(row=2, column=0)

        Button(
            self.control_panel,
            text='<< Prev',
            width=10,
            command=self.prev_image
        ).pack(side=LEFT, padx=5, pady=3)

        Button(
            self.control_panel,
            text='Next >>',
            width=10,
            command=self.next_image
        ).pack(side=LEFT, padx=5, pady=3)

        self.progress_label = Label(
            self.control_panel,
            text="Progress:     /    "
        )
        self.progress_label.pack(side=LEFT, padx=5)

        Label(
            self.control_panel,
            text="Go to Image No."
        ).pack(side=LEFT, padx=5)

        self.idx_entry = Entry(self.control_panel, width=5)
        self.idx_entry.pack(side=LEFT)

        Button(
            self.control_panel,
            text='Go',
            command=self.go_to_image
        ).pack(side=LEFT)

        # display mouse position
        self.mouse_pos_label = Label(
            self.control_panel,
            text=''
        )
        self.mouse_pos_label.pack(side=RIGHT)
        self.mouse_pos_label.config(
            text='x: %d, y: %d' % (0, 0)
        )

        Button(
            self,
            text='Auto Annotate',
            command=self.auto_annotate
        ).grid(row=3, column=0, padx=pad_x, pady=pad_y)

        Button(
            self,
            text='Generate ground truth',
            command=self.generate_ground_truth
        ).grid(row=4, column=0, padx=pad_x, pady=pad_y)

        # Step Navigation
        self.nav_panel = Frame(self)
        self.nav_panel.grid(row=5, column=0)

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

    def on_show_frame(self, _):
        if self.controller.working_folder is not None:
            self.load_images(self.controller.working_folder)

    def generate_ground_truth(self):
        ground_truths = []
        label_directory = self.controller.working_folder + "/labels"
        sorted_files = sorted(os.listdir(label_directory))
        for f in sorted_files:
            fh = open(label_directory + "/" + f, "r")
            lines = fh.read().split("\n")
            ground_truth = ",".join(lines[0].strip().split(" "))
            ground_truths.append(ground_truth)

        txt_outfile = open(self.controller.working_folder + "/" + "groundtruth_rect.txt", "w+")
        for i in ground_truths:
            txt_outfile.write(str(i) + "\n")
        txt_outfile.close()
        tkMessageBox.showinfo("Success", "Groundtruth is generated successfully.\n")

    def auto_annotate(self):
        self.controller.show_progress(True)
        folder = self.controller.working_folder
        self.images = natsort.natsorted(
            glob.glob(os.path.join(folder + "/img", '*.jpg')), reverse=False
        )
        if len(self.images) == 0:
            print 'No .JPG images found in the specified dir!'
            return

        # create labels directory
        self.label_directory = folder + "/labels"
        if not os.path.isdir(self.label_directory):
            os.mkdir(self.label_directory)

        print '%d images loaded from %s' % (self.total, folder)

        for image in self.images:
            image_name = os.path.split(image)[-1].split('.')[0]
            label_name = image_name + '.txt'
            self.label_file = os.path.join(self.label_directory, label_name)
            output = self.yolo.predict_location_from_image(image)
            if output:
                x = int(output[0])
                y = int(output[1])
                w = int(output[2]) // 2
                h = int(output[3]) // 2
                location = [x - w, y - h, x + w, y + h]
                with open(self.label_file, 'w') as f:
                    f.write(' '.join(map(str, location)) + '\n')
                print('Label saved for image:', image)
        self.controller.show_progress(False)

    def next_step(self):
        from ui.single_object_detection import SingleDetection
        self.controller.show_frame(SingleDetection)

    def prev_step(self):
        from ui.upload import Upload
        self.controller.show_frame(Upload)

    def mouse_click(self, event):
        if self.mouse['click'] == 0:
            self.clear_bbox()
            self.mouse['x'], self.mouse['y'] = event.x, event.y
        else:
            x1, x2 = min(self.mouse['x'], event.x), max(self.mouse['x'], event.x)
            y1, y2 = min(self.mouse['y'], event.y), max(self.mouse['y'], event.y)
            self.bbox = (x1, y1, x2, y2)
        self.mouse['click'] = 1 - self.mouse['click']

    def mouse_move(self, event):
        self.mouse_pos_label.config(
            text='x: %d, y: %d' % (event.x, event.y)
        )

        if self.tk_image:
            if self.hl:
                self.main_panel.delete(self.hl)
            self.hl = self.main_panel.create_line(
                0, event.y, self.tk_image.width(), event.y, width=2
            )
            if self.vl:
                self.main_panel.delete(self.vl)
            self.vl = self.main_panel.create_line(
                event.x, 0, event.x, self.tk_image.height(), width=2
            )

        if 1 == self.mouse['click']:
            if self.bbox_id:
                self.main_panel.delete(self.bbox_id)
            self.bbox_id = self.main_panel.create_rectangle(self.mouse['x'], self.mouse['y'],
                                                            event.x, event.y,
                                                            width=2,
                                                            outline=COLOR)

    def cancel_bbox(self):
        if 1 == self.mouse['click']:
            if self.bbox_id:
                self.main_panel.delete(self.bbox_id)
                self.bbox_id = None
                self.mouse['click'] = 0

    def prev_image(self):
        self.save_label()
        if self.cur > 1:
            self.cur -= 1
            self.load_image()

    def next_image(self):
        self.save_label()
        if self.cur < self.total:
            self.cur += 1
            self.load_image()

    def go_to_image(self):
        idx = int(self.idx_entry.get())
        if 1 <= idx <= self.total:
            self.save_label()
            self.cur = idx
            self.load_image()

    def save_label(self):
        if self.bbox is not None:
            with open(self.label_file, 'w') as f:
                f.write(' '.join(map(str, self.bbox)) + '\n')
            print 'Label No. %d saved' % self.cur

    def load_images(self, folder):
        self.images = natsort.natsorted(
            glob.glob(os.path.join(folder + "/img", '*.jpg')), reverse=False
        )

        if len(self.images) == 0:
            print 'No .JPG images found in the specified dir!'
            return

        self.cur = 1
        self.total = len(self.images)

        # create labels directory
        self.label_directory = folder + "/labels"
        if not os.path.isdir(self.label_directory):
            os.mkdir(self.label_directory)

        self.load_image()
        print '%d images loaded from %s' % (self.total, folder)

    def load_image(self):
        image_path = self.images[self.cur - 1]
        image = Image.open(image_path)
        self.tk_image = ImageTk.PhotoImage(image)
        self.main_panel.config(
            width=max(self.tk_image.width(), 256), height=max(self.tk_image.height(), 256)
        )
        self.main_panel.create_image(0, 0, image=self.tk_image, anchor=NW)
        self.progress_label.config(text="%04d/%04d" % (self.cur, self.total))

        # load labels
        self.clear_bbox()

        image_name = os.path.split(image_path)[-1].split('.')[0]
        label_name = image_name + '.txt'
        self.label_file = os.path.join(self.label_directory, label_name)
        if os.path.exists(self.label_file):
            with open(self.label_file) as f:
                for (i, line) in enumerate(f):
                    if line.strip() != "0":
                        tmp = [int(t.strip()) for t in line.split()]
                        self.bbox = tuple(tmp)
                        tmp_id = self.main_panel.create_rectangle(tmp[0], tmp[1],
                                                                  tmp[2], tmp[3],
                                                                  width=2,
                                                                  outline=COLOR)
                        self.bbox_id = tmp_id

    def clear_bbox(self):
        self.main_panel.delete(self.bbox_id)
        self.bbox = None
        self.bbox_id = None
