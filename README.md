# DL-ODT-for-UAV
Deep Learning based Object detection and tracking for UAV videos

# TODO
- [ ] Shell script to automate development setup
- [ ] Create mockup screens - https://drive.google.com/file/d/1yy86lZZ4gLDnyp_oiNmoiufngoSfyGVv/view?usp=sharing
- [ ] Kivy based high level GUI layout
- [ ] Allow users to load video files


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
