from Tkinter import *
import os
import natsort
import cv2
from learning.rolo import ROLO
sys.path.append("./rolo/utils")
import ROLO_utils as utils

title_font = ("Times New Roman", 18, "bold")
pad_x = 10
pad_y = 10


class SingleTracking(Frame):
    def __init__(self, master, controller, **kw):
        Frame.__init__(self, master, **kw)

        self.controller = controller
        self.grid_columnconfigure(0, weight=1)
        self.rolo_train = None

        Label(
            self, text="LSTM based Single Object Tracking", font=title_font
        ).grid(row=0, column=0)

        Label(
            self, text="Training The LSTM model"
        ).grid(row=1, column=0)

        self.container = Frame(self)
        self.container.grid(row=2, column=0, padx=pad_x, pady=pad_y)

        # Get Epoch Value
        Label(
            self.container, text="Enter No. Epoches for training"
        ).pack(side=LEFT)

        self.epoches_entry = Entry(self.container)
        self.epoches_entry.pack(side=LEFT)
        self.epoches_entry.insert(END, '2')

        Label(
            self, text="Train the model"
        ).grid(row=3, column=0, padx=pad_x, pady=pad_y)
        Button(
            self, text="Training", command=self.train
        ).grid(row=4, column=0, padx=pad_x, pady=pad_y)

        Label(
            self, text="Test the model"
        ).grid(row=5, column=0, padx=pad_x, pady=pad_y)
        Button(
            self, text="Testing", command=self.test
        ).grid(row=6, column=0, padx=pad_x, pady=pad_y)

        Label(self, text="Tracking").grid(row=7, column=0, padx=pad_x, pady=pad_y)
        global variable
        variable = StringVar(self)
        a = os.listdir("./rolo/data/")

        variable.set(a[0])
        OptionMenu(self, variable, *a).grid(row=8, column=0)
        Button(self, text="Ok", command=self.demo).grid(row=9, column=0, padx=pad_x, pady=pad_y)


        self.nav_panel = Frame(self)
        self.nav_panel.grid(row=10, column=0, sticky="s")
        self.grid_rowconfigure(10, weight=1)

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

    def prev_step(self):
        from ui.single_object_detection import SingleDetection
        self.controller.show_frame(SingleDetection)

    def next_step(self):
        from ui.references import References
        self.controller.show_frame(References)

    def train(self):
        if self.rolo_train is None:
            self.rolo_train = ROLO()
            self.rolo_train.train(int(self.epoches_entry.get()))
        else:
            self.rolo_train.train(int(self.epoches_entry.get()))

    def test(self):
        if self.rolo_train is None:
            self.rolo_train = ROLO()
            self.rolo_train.test()
        else:
            self.rolo_train.test()

    def demo(self):

        ''' PARAMETERS '''
        num_steps = 6
        wid = 500
        ht = 500
        #basepath = Path("./rolo/data/")
        total = 0
        rolo_avgloss = 0
        yolo_avgloss = 0

        b = variable.get()
        folder_path = os.path.dirname("./rolo/data/") + '/' + b + '/'

        img_fold_path = os.path.join(folder_path) + 'img/'

        gt_file_path = os.path.join(folder_path) + 'groundtruth_rect.txt'
        yolo_out_path = os.path.join(folder_path) + 'yolo_out/'
        rolo_out_path = os.path.join(folder_path) + 'rolo_out_train/'
        paths_imgs = utils.load_folder(img_fold_path)
        paths_imgs = natsort.natsorted(paths_imgs, reverse=False)
        paths_rolo = utils.load_folder(rolo_out_path)
        paths_rolo = natsort.natsorted(paths_rolo, reverse=False)
        lines = utils.load_dataset_gt(gt_file_path)

        # Define the codec and create VideoWriter object
        # fourcc= cv2.cv.CV_FOURCC(*'DIVX')
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video_name = 'test.avi'
        video_path = os.path.join('output/videos/', video_name)
        video = cv2.VideoWriter(video_path, fourcc, 20, (wid, ht))

        center_points_gt = []
        center_points_rolo = []

        for i in range(len(paths_rolo) - num_steps):
            id = i + 1
            test_id = id + num_steps - 2  # * num_steps + 1

            path = paths_imgs[test_id]
            img = utils.file_to_img(path)

            if (img is None): break

            yolo_location = utils.find_yolo_location(yolo_out_path, test_id)
            yolo_location = utils.locations_normal(wid, ht, yolo_location)
            # print(yolo_location)

            rolo_location = utils.find_rolo_location(rolo_out_path, test_id)
            rolo_location = utils.locations_normal(wid, ht, rolo_location)
            # print(rolo_location)

            gt_location = utils.find_gt_location(lines, test_id - 1)
            # gt_location= locations_from_0_to_1(None, 480, 640, gt_location)
            # gt_location = locations_normal(None, 480, 640, gt_location)
            # print('gt: ' + str(test_id))
            # print(gt_location)

            # calculate centroid
            # (x1 +x2)/2
            # (y1 +22)/2
            l_gt = gt_location
            center_x = l_gt[0] + (l_gt[2] / 2)
            center_y = l_gt[1] + (l_gt[3] / 2)
            center_points_gt.append((center_x, center_y))

            # todo, fix center calculation
            l_rolo = rolo_location
            # center_x = int(l_rolo[0] + (l_rolo[2] / 2))
            # center_y = int(l_rolo[1] + (l_rolo[3] / 2))
            center_x = int(l_rolo[0])
            center_y = int(l_rolo[1])
            center_points_rolo.append((center_x, center_y))

            frame = utils.debug_3_locations(
            img, gt_location, yolo_location, rolo_location,
            center_points_gt,
            center_points_rolo
            )
            video.write(frame)

            utils.createFolder(os.path.join('./rolo/output/frames/', b))
            frame_name = os.path.join('./rolo/output/frames/', b, str(test_id) + '.jpg')

        # print(frame_name)
            cv2.imwrite(frame_name, frame)
        # cv2.imshow('frame',frame)
        # cv2.waitKey(100)

            rolo_loss = utils.cal_rolo_IOU(rolo_location, gt_location)
            rolo_avgloss += rolo_loss
            yolo_loss = utils.cal_yolo_IOU(yolo_location, gt_location)
            yolo_avgloss += yolo_loss
            total += 1
        rolo_avgloss /= total
        yolo_avgloss /= total
        print("yolo_avg_iou = ", yolo_avgloss)
        print("rolo_avg_iou = ", rolo_avgloss)
        video.release()
        cv2.destroyAllWindows()
