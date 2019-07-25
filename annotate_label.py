import os
import natsort
import glob
imagepath = "./ROLO/DATA/car18/img"
label_path = os.path.dirname(imagepath)+"/"+"labels/"
for file in os.listdir(label_path):
	'''if file.endswith(".txt"):
       		label_txt= os.path.join(label_path, file)
		print(label_txt)'''
	orderedImageList = glob.glob(os.path.join(label_path, '*.txt'))
	orderedImageList = natsort.natsorted(orderedImageList, reverse=False)
	print(orderedImageList)
