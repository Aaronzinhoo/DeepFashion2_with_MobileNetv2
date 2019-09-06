########
# Convert Deep Fashion 2 Dataset to per image annotations xml format file
'''
File Directory to use this should look like

cwd -> DeepFashion2
    ->train
        ->image
        ->Annotations
            ...
    ->val
        ->image
        ->Annotations
            ...
'''
########

from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import sys
import os

top_categoryID = [1,2,3,4,5,6]
bottom_categoryID = [7,8,9]
full_categoryID = [10,11,12,13]


def main(dataset_type = 'train' , num_images=0):
    '''
            Convert json annotations to xml annotations for ImageNetv2.0. We decide to keep the label,bnd boxes,image id, category id
            among a couple other features that can be disregarded depending on your purpose. 
                Args:
                dataset_type: 
                    'train', 'val','test' determines which data set is having its properties converted to xml form
                num_images:
                        for testing purposes and to work on smaller subsets of the dataset
    '''
    annotations_dir = 'Annotations'
    if num_images == 0:
        if dataset_type == 'val':
            num_images = 32153
        elif dataset_type == 'train':
            num_images = 191961
        elif dataset_type  =='test':
            num_images=60000
        else:
            raise RuntimeError('not valid dataset type')
            exit()

    anno_path = os.path.join(dataset_type,annotations_dir)
    if not tf.io.gfile.exists(anno_path):
        tf.io.gfile.mkdir(anno_path)
    
    dataset = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    cat_count = {'top':0 , 'bottom':0, 'full':0}
    #num_images = 9
    sub_index = 0 # the index of ground truth instance
    print('Converting Image Features to XML')
    for num in range(1,num_images+1):
        if (num_images-1)%50000==0:
            #print('Converted Images = %s' % str(num))
        dataset['annotations'] = []
        json_name = '../'+dataset_type+'/annos/' + str(num).zfill(6)+'.json'
        image_name = '../'+dataset_type+'/image/' + str(num).zfill(6)+'.jpg'

        if (num>=0):
            imag = Image.open(image_name)
            width, height = imag.size
            with open(json_name, 'r') as f:
                temp = json.loads(f.read())
                pair_id = temp['pair_id']

                dataset['images'].append({
                    'coco_url': '',
                    'date_captured': '',
                    'file_name': str(num).zfill(6) + '.jpg',
                    'flickr_url': '',
                    'id': num,
                    'license': 0,
                    'width': width,
                    'height': height
                })
                for i in temp:
                    if i == 'source' or i=='pair_id':
                        continue
                    else:
                        #points = np.zeros(294 * 3)
                        sub_index = sub_index + 1
                        box = temp[i]['bounding_box']
                        bbox={'xmin':box[0], 'ymin':box[1], 'xmax':box[2],'ymax':box[3]}
                        cat = temp[i]['category_id']
                        if cat in top_categoryID:
                            name='top'
                        elif cat in bottom_categoryID:
                            name = 'botttom'
                        elif cat in full_categoryID:
                            name = 'full'
                        else:
                            raise ValueError('category ID not found')
                        cat_count +=1
                        style = temp[i]['style']
                        seg = temp[i]['segmentation']
                        landmarks = temp[i]['landmarks']

                        
                        dataset['annotations'].append({
                            'name': name,
                            #'area': w*h,
                            'bbox': bbox,
                            'category_id': cat,
                            'id': sub_index,
                            #'pair_id': pair_id,
                            'image_id': num,
                            #'iscrowd': 0,
                            #'style': style,
                            #'num_keypoints':num_points,
                            #'keypoints':points.tolist(),
                            #'segmentation': seg,
                        }
                        )            
                #write json to xml and store it in Annotations folder
                anno_to_xml = dicttoxml(dataset['annotations'],custom_root='annotation',attr_type =False)
                anno_to_xmlpretty = parseString(anno_to_xml).toprettyxml()
                with open((dataset_type +'/Annotations/%s%s' % (str(num).zfill(6),'.xml')) , 'w') as f:
                    f.write(anno_to_xmlpretty)   
    print('Your Category Amounts Are: ',  cat_count)

if __name__ == '__main__':
    #if not len(sys.argv == 3):
    #raise RuntimeError('format: make_deepfashion_XMLannotations.py <dir here> <data set type>')
    #exit()
    if len(sys.argv) ==1:
        main()
    elif len(sys.argv) ==2:
        main(sys.argv[1])
    elif len(sys.argv) ==3:
        main(sys.argv[1],sys.argv[2])
    else:
        print('Must provide no greater than 2 argument <dataset to transer>')
        exit()

