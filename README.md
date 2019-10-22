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
   `pip install pandas'
   

6. To install the dependencies,
    `pip install -r requirements.txt`

7. Make the run.py file executable
    `chmod +x app.py`
    
8. Download the yolo weight file from https://drive.google.com/open?id=1BAxKaRz-qkp4SZwY_xePFKsJ1HOADTq1 and place it at /DL-ODT-for-UAV/rolo/weights/

9. Download https://drive.google.com/open?id=1R3_mUWD_tzLt2jBeakWdiZei9NJldxJB and place it at /DL-ODT-for-UAV/rolo/output/rolo_model/
`https://drive.google.com/drive/folders/1jwlw4kfceFfYQvJixKGZNQJ-_R8vihpF?usp=sharing`


8. install `sudo apt install protobuf-compiler` , `pip install pillow`,`pip install lxml`,` pip install Cython`,`pip install contextlib2`,`pip install jupyter`,`pip install matplotlib`,`pip install pandas`, `pip install opencv-python`

9. set the path by `export PYTHONPATH=/home/ancy/PycharmProjects/DL-ODT-for-UAV/models:/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research:/home/ancy/PycharmProjects/DL-ODT-for-UAV/models/research/slim:/usr/bin`

10. cd models, cd research leads to DL-ODT-for-UAV/model/research/   

11. compile the protobuf files `protoc --python_out=. ./object_detection/protos/anchor_generator.proto ./object_detection/protos/argmax_matcher.proto ./object_detection/protos/bipartite_matcher.proto ./object_detection/protos/box_coder.proto ./object_detection/protos/box_predictor.proto ./object_detection/protos/eval.proto ./object_detection/protos/faster_rcnn.proto ./object_detection/protos/faster_rcnn_box_coder.proto ./object_detection/protos/grid_anchor_generator.proto ./object_detection/protos/hyperparams.proto ./object_detection/protos/image_resizer.proto ./object_detection/protos/input_reader.proto ./object_detection/protos/losses.proto ./object_detection/protos/matcher.proto ./object_detection/protos/mean_stddev_box_coder.proto ./object_detection/protos/model.proto ./object_detection/protos/optimizer.proto ./object_detection/protos/pipeline.proto ./object_detection/protos/post_processing.proto ./object_detection/protos/preprocessor.proto ./object_detection/protos/region_similarity_calculator.proto ./object_detection/protos/square_box_coder.proto ./object_detection/protos/ssd.proto ./object_detection/protos/ssd_anchor_generator.proto ./object_detection/protos/string_int_label_map.proto ./object_detection/protos/train.proto ./object_detection/protos/keypoint_box_coder.proto ./object_detection/protos/multiscale_anchor_generator.proto ./object_detection/protos/graph_rewriter.proto ./object_detection/protos/calibration.proto ./object_detection/protos/flexible_grid_anchor_generator.proto`
This creates a name_pb2.py file from every name.proto file in the models/research/object_detection/protos folder.

12. Run the following commands from the models/research/ directory: `python setup.py build` `python setup.py install`

   
8. To run,
    `./app.py`
9. To deactivate,
    `deactivate`
    

 
