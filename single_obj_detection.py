from Tkinter import *
import Tkinter as tk
from YOLO_network import YOLO_TF


LARGE_FONT = ("Verdana", 12)


class Single_Obj_Detection(tk.Frame):
    def __init__(self, master, controller):
        Frame.__init__(self, master)

        self.controller = controller
        self.YOLO_TF = YOLO_TF()

        self.pageTitle = Label(self, text="Single Object Detection", font=LARGE_FONT)
        self.pageTitle.grid(row=0, column=0, columnspan=2)

        self.browse_button1 = Button(self, text="Prepare training data", command = self.YOLO_TF)
        self.browse_button1.grid(row=1, column=1, sticky="we")

        self.browse_button2 = Button(self, text="Show Detection")
        self.browse_button2.grid(row=2, column=1, sticky="we")


if __name__ == '__main__':
    root = Tk()


    # window properties
    root.geometry('1280x800')
    root.resizable(width=True, height=True)

    tool = Single_Obj_Detection(root)
    tool.grid(row=0, column=0, sticky="nsew")

    root.mainloop()
