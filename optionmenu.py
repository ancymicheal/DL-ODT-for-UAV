from Tkinter import *
import os
master = Tk()
#creat a list of folder name


variable = StringVar(master)
a = os.listdir("./ROLO/DATA/")
variable.set(a[0])
OptionMenu(master, variable, *a).pack()
def ok():
    b = variable.get()

    folder_path = os.path.dirname("./ROLO/DATA/") + '/'+ b + '/'
    img_fold_path = os.path.join(folder_path)+'img/'
    gt_file_path = os.path.join(folder_path)+'groundtruth_rect.txt'
    yolo_out_path = os.path.join(folder_path)+ 'yolo_out/'
    rolo_out_path = os.path.join(folder_path)+ 'rolo_out_train/'
    print(img_fold_path)
    print(gt_file_path)
    print(yolo_out_path)
    print(rolo_out_path)
    master.quit()

button = Button(master, text="OK", command=ok)
button.pack()

mainloop()


#add the list to dorpdown
#creat ok buttn
#click on button play demo

'''from Tkinter import *

master = Tk()

var = StringVar(master)
var.set("one") # initial value

option = OptionMenu(master, var, "one", "two", "three", "four")
option.pack()


def ok():
    print "value -", var.get()
    master.quit()

button = Button(master, text="OK", command=ok)
button.pack()

mainloop()'''
