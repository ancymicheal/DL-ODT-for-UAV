import os
from os import walk, getcwd
import re

classes = ["boat6"]

cls = "boat6"

workingDirectory = "BBox-Label-Tool-master"
mypath = "/home/ancy/ROLO/" + workingDirectory + "/Labels/" + cls + "/"
outpath = "/home/ancy/ROLO/" + workingDirectory + "/labels-yolo/" + cls + "/"

ground_truths = []

# Get the list of files
wd = getcwd()
list_file = open('%s/%s/%s_list.txt'%(wd, workingDirectory, cls), 'w')
txt_name_list = []
for (dirpath, dirnames, filenames) in walk(mypath):
    txt_name_list.extend(filenames)
    break
print(txt_name_list)


# sort the file list
sorted_list = sorted(txt_name_list)


# read contents of each file
for file in sorted_list:
	txt_path = mypath + file
	fh = open(txt_path, "r")
	
	lines = fh.read().split("\n")
	#del lines[0]
        #print(lines[0])
        #print(lines[1])
	#del lines[1]

	ground_truth = re.sub("\s+", ",", lines[1].strip())
	ground_truths.append(ground_truth)


txt_outfile = open(outpath + "groundtruth_rect.txt", "w+")
for i in ground_truths:
	txt_outfile.write(str(i) + "\n")

txt_outfile.close()
