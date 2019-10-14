from Tkinter import *

title_font = ("Times New Roman", 18, "bold")
pad_x = 10
pad_y = 10


class DTType(Frame):
    def __init__(self, master, controller, **kw):
        Frame.__init__(self, master, **kw)
        self.controller = controller

        self.grid_columnconfigure(0, weight=1)

        # Title
        self.page_title = Label(
            self,
            text="Select Object Detection And Tracking Type",
            font=title_font,
        )
        self.page_title.grid(row=0, column=0, sticky='n', padx=pad_x, pady=pad_y)

        Button(
            self,
            text="Single Object Detection and Tracking",
            borderwidth=4,
            width=50,
            command=self.select_single
        ).grid(row=1, column=0, padx=pad_x, pady=pad_y)

        Button(
            self,
            text="Multiple Object Detection (YET TO BE IMPLEMENTED)",
            borderwidth=4,
            width=50,
	    command=self.select_multiple
        ).grid(row=2, column=0, padx=pad_x, pady=pad_y)

    def select_single(self):
        from ui.upload import Upload
        self.controller.show_frame(Upload)

    def select_multiple(self):
        from ui.upload_mul import UploadMul
        self.controller.show_frame(UploadMul)

