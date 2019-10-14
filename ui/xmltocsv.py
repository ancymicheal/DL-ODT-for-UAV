import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
image_path = "./models/research/object_detection/images/"
for folder in ['train','test']:
	image_path_folder = os.path.join(image_path + folder)
	
	xml_list = []
    	for xml_file in glob.glob(image_path_folder + '/*.xml'):
        	tree = ET.parse(xml_file)
        	root = tree.getroot()
        	for member in root.findall('object'):
            		value = (root.find('filename').text,
                     		int(root.find('size')[0].text),
                     		int(root.find('size')[1].text),
                     		member[0].text,
                     		int(member[4][0].text),
                     		int(member[4][1].text),
                     		int(member[4][2].text),
                     		int(member[4][3].text)
                     		)
            		xml_list.append(value)
    	column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    	xml_df = pd.DataFrame(xml_list, columns=column_name)
        xml_df.to_csv((image_path + folder + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')


