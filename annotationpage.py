import tkMessageBox
from Tkinter import *
import Tkinter as tk
import os
import tkFileDialog as filedialog
import cv2
from PIL import Image, ImageTk
import PIL
import glob

LARGE_FONT = ("Verdana", 12)

# colors for the bboxes
COLORS = ['red', 'blue', 'yellow', 'pink', 'cyan', 'green', 'black']
# image sizes for the examples
SIZE = 256, 256


class AnnotationPage(tk.Frame):
    def __init__(self, master, controller):
        Frame.__init__(self, master)

        self.controller = controller

        self.bind("<Expose>", self.print_event)

        # Title
        self.pageTitle = Label(self, text="Annotation Page", font=LARGE_FONT)
        self.pageTitle.grid(row=0, column=0)

        # initialize global state
        self.imageDir = ''
        self.imageList = []
        self.egDir = ''
        self.egList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.labelfilename = ''
        self.tkimg = None

        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.hl = None
        self.vl = None

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self, text="Image Dir:")
        self.label.grid(row=0, column=0, sticky=E)
        self.entry = Entry(self)
        self.entry.grid(row=0, column=1, sticky=W + E)
        self.ldBtn = Button(self, text="Load", command=self.loadDir)
        self.ldBtn.grid(row=0, column=2, sticky=W + E)

        # main panel for labeling
        self.mainPanel = Canvas(self, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouseClick)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.bind("<Escape>", self.cancelBBox)  # press <Espace> to cancel current bbox
        self.bind("s", self.cancelBBox)
        self.bind("a", self.prevImage)  # press 'a' to go backforward
        self.bind("d", self.nextImage)  # press 'd' to go forward
        self.mainPanel.grid(row=1, column=1, rowspan=4, sticky=W + N)

        # showing bbox info & delete bbox
        self.lb1 = Label(self, text='Bounding boxes:')
        self.lb1.grid(row=1, column=2, sticky=W + N)
        self.listbox = Listbox(self, width=22, height=12)
        self.listbox.grid(row=2, column=2, sticky=N)
        self.btnDel = Button(self, text='Delete', command=self.delBBox)
        self.btnDel.grid(row=3, column=2, sticky=W + E + N)
        self.btnClear = Button(self, text='ClearAll', command=self.clearBBox)
        self.btnClear.grid(row=4, column=2, sticky=W + E + N)

        # control panel for image navigation
        self.ctrPanel = Frame(self)
        self.ctrPanel.grid(row=5, column=1, columnspan=2, sticky=W + E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width=10, command=self.prevImage)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width=10, command=self.nextImage)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        self.progLabel = Label(self.ctrPanel, text="Progress:     /    ")
        self.progLabel.pack(side=LEFT, padx=5)
        self.tmpLabel = Label(self.ctrPanel, text="Go to Image No.")
        self.tmpLabel.pack(side=LEFT, padx=5)
        self.idxEntry = Entry(self.ctrPanel, width=5)
        self.idxEntry.pack(side=LEFT)
        self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoImage)
        self.goBtn.pack(side=LEFT)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side=RIGHT)

        self.columnconfigure(1, weight=1)
        self.rowconfigure(4, weight=1)

        # for debugging
        ##        self.setImage()
        ##        self.loadDir()

    def print_event(self, event):
        position = "(x={}, y={})".format(event.x, event.y)
        print(event.type, "event", position)
        if self.imageDir == '':
            self.loadDir(self.controller.get_frame_directory())

    def loadDir(self, image_dir):

        print("loading images..." + image_dir)

        # if not dbg:
        #     s = self.entry.get()
        #     self.parent.focus()
        #     self.category = int(s)
        # else:
        #     # s = r'D:\workspace\python\labelGUI'
        #     s = r'/home/ancymicheal/ROLO/BBox-Label-Tool-master/Images'

        # self.imageDir = os.path.join(r'/home/ancymicheal/ROLO/BBox-Label-Tool-master/Images', '%03d'
        # % (self.category))
        self.imageDir = image_dir
        self.unorderedImageList = sorted(glob.glob(os.path.join(self.imageDir, '*.jpg')))

        self.imageList = [""] * len(self.unorderedImageList)

        print(self.unorderedImageList)

        # todo, sort image list
        # for i in self.unorderedImageList:
        #     self.imageList[int(i[58:-4])] = i
        self.imageList = self.unorderedImageList

        print(self.imageList)
        if len(self.imageList) == 0:
            print 'No .JPG images found in the specified dir!'
            return

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)

        # create labels directory
        self.label_dir = self.controller.get_frame_directory() + "/labels"
        if not os.path.isdir(self.label_dir):
            os.mkdir(self.label_dir)

        # set up output dir
        # self.outDir = os.path.join(r'/home/ancymicheal/ROLO/BBox-Label-Tool-master/Labels', '%03d' % (self.category))
        # if not os.path.exists(self.outDir):
        #     os.mkdir(self.outDir)

        self.outDir = self.label_dir

        # load example bboxes
        # self.egDir = os.path.join(r'/home/ancymicheal/ROLO/BBox-Label-Tool-master/Images', '%03d' % (self.category))
        # if not os.path.exists(self.egDir):
        #     return

        self.loadImage()
        print '%d images loaded from %s' % (self.total, self.imageDir)

    def loadImage(self):
        # load image
        imagepath = self.imageList[self.cur - 1]
        print(self.cur)
        print(imagepath)
        self.img = Image.open(imagepath)
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width=max(self.tkimg.width(), 400), height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
        self.progLabel.config(text="%04d/%04d" % (self.cur, self.total))

        # load labels
        self.clearBBox()
        self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        labelname = self.imagename + '.txt'
        self.labelfilename = os.path.join(self.outDir, labelname)
        bbox_cnt = 0
        if os.path.exists(self.labelfilename):
            with open(self.labelfilename) as f:
                for (i, line) in enumerate(f):
                    if i == 0:
                        bbox_cnt = int(line.strip())
                        continue
                    tmp = [int(t.strip()) for t in line.split()]
                    ##                    print tmp
                    self.bboxList.append(tuple(tmp))
                    tmpId = self.mainPanel.create_rectangle(tmp[0], tmp[1], \
                                                            tmp[2], tmp[3], \
                                                            width=2, \
                                                            outline=COLORS[(len(self.bboxList) - 1) % len(COLORS)])
                    self.bboxIdList.append(tmpId)
                    self.listbox.insert(END, '(%d, %d) -> (%d, %d)' % (tmp[0], tmp[1], tmp[2], tmp[3]))
                    self.listbox.itemconfig(len(self.bboxIdList) - 1,
                                            fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])

    def saveImage(self):
        with open(self.labelfilename, 'w') as f:
            f.write('%d\n' % len(self.bboxList))
            for bbox in self.bboxList:
                f.write(' '.join(map(str, bbox)) + '\n')
        print 'Image No. %d saved' % (self.cur)

    def mouseClick(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
        else:
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            self.bboxList.append((x1, y1, x2, y2))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.listbox.insert(END, '(%d, %d) -> (%d, %d)' % (x1, y1, x2, y2))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
        self.STATE['click'] = 1 - self.STATE['click']

    def mouseMove(self, event):
        self.disp.config(text='x: %d, y: %d' % (event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width=2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width=2)
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'], \
                                                          event.x, event.y, \
                                                          width=2, \
                                                          outline=COLORS[len(self.bboxList) % len(COLORS)])

    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def delBBox(self):
        sel = self.listbox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.listbox.delete(idx)

    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []

    def prevImage(self, event=None):
        self.saveImage()
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def nextImage(self, event=None):
        self.saveImage()
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.saveImage()
            self.cur = idx
            self.loadImage()


if __name__ == '__main__':
    root = Tk()

    # window properties
    root.geometry('1280x800')
    root.resizable(width=True, height=True)

    tool = AnnotationPage(root)
    tool.grid(row=0, column=0, sticky="nsew")

    root.mainloop()
