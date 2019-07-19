# DL-ODT-for-UAV
Deep Learning based Object detection and tracking for UAV videos
- UBUNTU

# TODO
- [ ] Create mockup screens - https://drive.google.com/file/d/1yy86lZZ4gLDnyp_oiNmoiufngoSfyGVv/view?usp=sharing
- [ ] fix canvas if possible


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
6. Install tensorflow 0.8  
python2.7 -m pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
7. Make the run.py file executable
    `chmod +x run.py`
8. To run,
    `./run.py`
9. To deactivate,
    `deactivate`
 
