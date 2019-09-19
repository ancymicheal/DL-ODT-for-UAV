# DL-ODT-for-UAV
Deep Learning based Object detection and tracking for UAV videos
- UBUNTU

# TODO
- [ ] Create mockup screens - https://drive.google.com/file/d/1yy86lZZ4gLDnyp_oiNmoiufngoSfyGVv/view?usp=sharing



# Steps
1. Make sure "pipenv" and "virtualenv" is installed
    `pip install --user pipenv`
    `pip install virtualenv` or `sudo apt install virtualenv`
    
2. cd into the DL-ODT-for-UAV folder

3. create a new virtual environment 
    `virtualenv -p /usr/bin/python2.7 venv`
4. Activate the environment
    `source venv/bin/activate`
    
5. Install tensorflow .
   `pip install tensorflow`
   `sudo apt-get install python-tk`
   

6. To install the dependencies,
    `pip install -r requirements.txt`

7. Make the run.py file executable
    `chmod +x app.py`
    
8. Download the yolo weight file from https://drive.google.com/open?id=1BAxKaRz-qkp4SZwY_xePFKsJ1HOADTq1 and place it at /DL-ODT-for-UAV/rolo/weights/

9. Download https://drive.google.com/open?id=1R3_mUWD_tzLt2jBeakWdiZei9NJldxJB and place it at /DL-ODT-for-UAV/rolo/output/rolo_model/
`https://drive.google.com/drive/folders/1jwlw4kfceFfYQvJixKGZNQJ-_R8vihpF?usp=sharing`
   
8. To run,
    `./app.py`
9. To deactivate,
    `deactivate`
 
