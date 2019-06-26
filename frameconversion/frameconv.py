import numpy as np
import cv2
from Tkinter import *
root = Tk()

def video1():
	cap = cv2.VideoCapture('/home/ashok/Data/videos/boat5.avi')
	while(cap.isOpened()):
		ret, frame = cap.read()
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

def frameconv():
	vidObj = cv2.VideoCapture('/home/ashok/Data/videos/boat5.avi')
	count = 0
	success = 1
	while success:
		#function extracts frame
		success, image = vidObj.read()
		cv2.imwrite("/home/ancymicheal/ROLO/frameconversion/000%d.jpg" %count, image)
		count +=1


button1 = Button(root, text="video1",command = video1).pack()
button2 = Button(root, text="frame conversion", command = frameconv).pack()

root.mainloop()


