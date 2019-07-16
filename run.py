#!venv/bin/python
import Tkinter as tk
from Tkinter import *

#
from annotationpage import AnnotationPage
from startpage import StartPage
from uploadvideopage import UploadVideoPage
from single_obj_detection import Single_Obj_Detection
from single_obj_tracking import Single_Obj_Tracking

title = "Object Detection And Tracking"
mainFrameBorder = "black"
footerFrameBorder = "red"

LARGE_FONT = ("Verdana", 12)
COLORS = ['red', 'blue', 'yellow', 'pink', 'cyan', 'green', 'black']
SIZE = 256, 256

steps = [
    'steps',
    'odt_type',
    'upload_video',
    'annotate',
    'single_obj_detection',
    'single_obj_tracking'
]
stepIndex = 0
currentStep = steps[stepIndex]


class ODTTool:
    def __init__(self, master):

        # Initialize Variables
        self.parent = master
        self.parent.title(title)
        self.frames = {}
        self.frames_dir = ""

        # Main Frame - Main GUI Content
        self.mainFrame = Frame(
            self.parent,
            highlightbackground=mainFrameBorder,
            highlightcolor=mainFrameBorder,
            highlightthickness=1
        )

        # Footer Frame - Prev and Next Buttons
        self.footerFrame = Frame(
            self.parent,
            highlightbackground=footerFrameBorder,
            highlightcolor=footerFrameBorder,
            highlightthickness=1
        )

        # Add both frames
        self.mainFrame.pack(fill=BOTH, expand=1)
        self.footerFrame.pack(fill=BOTH, expand=1)

        for F in (StartPage, OdtTypePage, UploadVideoPage, AnnotationPage, Single_Obj_Detection, Single_Obj_Tracking):
            frame = F(self.mainFrame, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # show current step in Main Frame
        self.show_main_frame_contents()

        # Show Footer Frame contents
        self.show_footer_frame_contents()

    def set_frame_directory(self, frames_dir):
        print("setting frames dir.." + frames_dir)
        self.frames_dir = frames_dir

    def get_frame_directory(self):
        return self.frames_dir

    def show_main_frame_contents(self):
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def go_to_previous_step(self):
        global stepIndex
        global currentStep

        if stepIndex > 0:
            stepIndex -= 1
            currentStep = steps[stepIndex]
            self.show_current_step_frame()

    def go_to_next_step(self):
        global stepIndex
        global currentStep

        if stepIndex < (len(steps) - 1):
            stepIndex += 1
            currentStep = steps[stepIndex]
            self.show_current_step_frame()

    def show_current_step_frame(self):
        print("loading frame...")
        if currentStep == 'steps':
            self.show_frame(StartPage)
        elif currentStep == 'odt_type':
            self.show_frame(OdtTypePage)
        elif currentStep == "upload_video":
            self.show_frame(UploadVideoPage)
        elif currentStep == "annotate":
            self.show_frame(AnnotationPage)
        elif currentStep == "single_obj_detection":
            self.show_frame(Single_Obj_Detection)
        elif currentStep == "single_obj_tracking":
            self.show_frame(Single_Obj_Tracking)
        else:
            self.show_frame(StartPage)

    def show_footer_frame_contents(self):
        Button(
            self.footerFrame, text="Previous", command=self.go_to_previous_step
        ).grid(row=0, column=0, sticky=W)

        Button(
            self.footerFrame, text="Next", command=self.go_to_next_step
        ).grid(row=0, column=1, sticky=E)


class OdtTypePage(tk.Frame):
    def __init__(self, master, controller):
        Frame.__init__(self, master)

        self.controller = controller

        # Title
        self.pageTitle = Label(self, text="ODT Type page", font=LARGE_FONT)
        self.pageTitle.grid(row=0, column=0)

        label = Label(self, text="Object Detection and Tracking", font=LARGE_FONT)
        label.grid(row=0, column=0, sticky=W + E)

        button1 = Button(self, text="Single Object Detection and Tracking", borderwidth=4, width=50,
                         command = self.select_single)
        button1.grid(row=1, column=0, sticky=W + E)

        button2 = tk.Button(self, text="Multiple Object Detection", borderwidth=4, width=50, )
        button2.grid(row=2, column=0, sticky=W + E)

    def select_single(self):
        global currentStep
        global steps
        currentStep = steps[2]
        print(currentStep)
        self.controller.show_current_step_frame()


if __name__ == '__main__':
    root = Tk()

    # window properties
    root.geometry('1280x800')
    root.resizable(width=True, height=True)

    tool = ODTTool(root)
    root.mainloop()
