import os
imagepath = "./ROLO/car18/img"
label_path = os.path.dirname(imagepath)+"/"+"labels/"
print(label_path)
for file in os.listdir(label_path):
	if file.endswith(".txt"):
       		print(os.path.join(label_path, file))