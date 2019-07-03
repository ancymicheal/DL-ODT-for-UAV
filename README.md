# DL-ODT-for-UAV
Deep Learning based Object detection and tracking for UAV videos

# TODO
- [ ] Create mockup screens - https://drive.google.com/file/d/1yy86lZZ4gLDnyp_oiNmoiufngoSfyGVv/view?usp=sharing


- [ ] place converted frames at videofilename(folder)->img(folder)->converted frames
- [ ] during annotation images should be read in order
- [ ] place annotated .txt at  videofilename(folder)->labels(folder)->individual frameslabels(txtfiles)
- [ ] place groundtruth at videofilename(folder)->groundtruth_rect.txt
- [ worked] for single obj detection place ROLO->weights-> YOLO_small.ckpt and run DL-ODT-for-UAV->YOLO_network 
- [ ] run training . and demo


# Steps
1. Make sure "pipenv" and "virtualenv" is installed
    `pip install --user pipenv`
    `pip install virtualenv`
2. cd into the DL-ODT-for-UAV folder
3. create a new virtual environment 
    `virtualenv -p /usr/bin/python2.7 venv`
4. Activate the environment
    `source venv/bin/activate`
5. To install the dependencies,
    `pip install -r requirements.txt`
6. Make the run.py file executable
    `chmod +x run.py`
7. To run,
    `./run.py`
8. To deactivate,
    `deactivate`
9. Install tensorflow 0.8  
python2.7 -m pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
