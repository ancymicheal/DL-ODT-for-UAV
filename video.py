import cv2 as cv
import os, sys
video_object = cv.VideoCapture("/home/ancy/Videos/boat2.avi")
frames_train = "/home/ancy/Videos/frames_train"
frames_test = "/home/ancy/Videos/frames_test"
frame_resize_dim = (int(500), int(500))
counter = 1
file_num = 1
success = 1
percentage_test = 20
index_test = round(100 / percentage_test)
while success:
	success, image = video_object.read()
	if success:
		re_sized = cv.resize(image, frame_resize_dim, interpolation=cv.INTER_AREA)
		if counter == index_test:
			counter = 1
			cv.imwrite(frames_test + "/000%d.jpg" % file_num, re_sized)
		else:
			cv.imwrite(frames_train + "/000%d.jpg" % file_num, re_sized)
			counter = counter + 1
	file_num = file_num+1
