#!/usr/bin/env python2
import Tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time

class App:
	def __init__(self, window, video_source=0):
		self.window = window
		self.video_source = '/home/ancymicheal/gui/boat5.avi'
	# open video surce
		self.vid = MyVideoCapture(self.video_source)
	
	#create a canvas
		self.canvas = Tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
		self.canvas.pack()

		self.btn_frameconv = Tkinter.Button(window, text ="Frame Conversion", width = 50, command = self.frameconv)
		self.btn_frameconv.pack(anchor=Tkinter.CENTER, expand= True)
		self.delay =15
		self.update()
		self.window.mainloop()


	def frameconv(self):

		vidObj = cv2.VideoCapture(self.video_source)
		count =0
		success = 1
		while success:
			#function extractes frame
			success, image = vidObj.read()
			cv2.imwrite("/home/ancymicheal/gui/frame conversion/frame%d.jpg" %count, image)
			count +=1

	def update(self):
		#get a frame from video source
		ret, frame = self.vid.get_frame()
		if ret:
			self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
			self.canvas.create_image(0, 0, image = self.photo, anchor = Tkinter.NW)
		self.window.after(self.delay, self.update)

class MyVideoCapture:
        def __init__(self, video_source=0):
        # Open the video source
		self.vid = cv2.VideoCapture(video_source)
		if not self.vid.isOpened():
			raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
		self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
	
	def get_frame(self):
		if self.vid.isOpened():
            		ret, frame = self.vid.read()
            		if ret:
                # Return a boolean success flag and the current frame converted to BGR
                		return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            		else:
                		return (ret, None)
       		else:
            		return (ret, None)

    # Release the video source when the object is destroyed
    	def __del__(self):
        	if self.vid.isOpened():
            		self.vid.release()

# Create a window and pass it to the Application object
App(Tkinter.Tk(), "Tkinter and OpenCV")
