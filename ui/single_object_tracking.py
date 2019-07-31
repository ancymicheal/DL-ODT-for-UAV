from Tkinter import *

from learning.rolo import ROLOTrain

title_font = ("Times New Roman", 18, "bold")
pad_x = 10
pad_y = 10


def test():
    ROLOTrain().test()


class SingleTracking(Frame):
    def __init__(self, master, controller, **kw):
        Frame.__init__(self, master, **kw)

        self.controller = controller
        self.grid_columnconfigure(0, weight=1)

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

        Button(
            self.container, text="Ok", command=self.get_epoches
        ).pack(side=LEFT)

        Label(
            self, text="Train the model"
        ).grid(row=3, column=0, padx=pad_x, pady=pad_y)
        Button(
            self, text="Training", command=train
        ).grid(row=4, column=0, padx=pad_x, pady=pad_y)

        Label(
            self, text="Test the model"
        ).grid(row=5, column=0, padx=pad_x, pady=pad_y)
        Button(
            self, text="Testing", command=test
        ).grid(row=6, column=0, padx=pad_x, pady=pad_y)

        self.nav_panel = Frame(self)
        self.nav_panel.grid(row=7, column=0, sticky="s")
        self.grid_rowconfigure(7, weight=1)

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

    def get_epoches(self):
        pass

    def train(self):
        ROLOTrain().train(self.epoches_entry.get())
