import Tkinter as tk


LARGE_FONT= ("Verdana", 12)


class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
	self.width =1000
	self.height =1000
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
	
        container.pack( side="top", fill="both", expand = True,)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, PageThree, PageFour):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="ODT in UAV videos", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        label = tk.Label(self, text="Steps", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = tk.Button(self, text="Next",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

  #      button2 = tk.Button(self, text="Visit Page 2",
   #                         command=lambda: controller.show_frame(PageTwo))
    #    button2.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Object Detection and Tracking", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

	button1 = tk.Button(self, text="Single Object Detection and Tracking",
                            command=lambda: controller.show_frame(PageTwo))
        button1.pack()

	button2 = tk.Button(self, text="Multiple Object Detection")
        button2.pack()

        button3 = tk.Button(self, text="Prev",
                            command=lambda: controller.show_frame(StartPage))
        button3.pack()

        button4 = tk.Button(self, text="Next",
                           command=lambda: controller.show_frame(PageTwo))
        button4.pack()


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Upload Video", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Prev",
                            command=lambda: controller.show_frame(PageOne))
        button1.pack()

        button2 = tk.Button(self, text="Next",
                            command=lambda: controller.show_frame(PageThree))
        button2.pack()
        
class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Frame Conversion", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Prev",
                            command=lambda: controller.show_frame(PageTwo))
        button1.pack()

        button2 = tk.Button(self, text="Next",
                            command=lambda: controller.show_frame(PageFour))
        button2.pack()

class PageFour(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Annotation", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Prev",
                            command=lambda: controller.show_frame(PageThree))
        button1.pack()

        button2 = tk.Button(self, text="Next")
        button2.pack()

app = SeaofBTCapp()
app.geometry("1000x1000")
app.mainloop()
