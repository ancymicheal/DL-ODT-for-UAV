# DL-ODT-for-UAV
Deep Learning based Object detection and tracking for UAV videos

# TODO
- [ ] Create mockup screens - https://drive.google.com/file/d/1yy86lZZ4gLDnyp_oiNmoiufngoSfyGVv/view?usp=sharing
- [ ] https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/ object trajectory has to be drawn. in the demo the trajectory of the object has to drawn  for green box as given in the link (IMPORTANT)
- [ ] run training , testing and demo
- [ ] pop up after training is complete


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
