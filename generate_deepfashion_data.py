import glob
import sys
import os
import xml.etree.ElementTree as ET
from random import random

def main(dataset_type = 'train',test_size = 0):
    '''
        Args
        filename: name of file that contains the labels for the classes
        dataset_type: provides which directory to grab images from
    '''
    filename = 'deepfashion-classes.txt'
    if test_size == 0:
        if dataset_type == 'val':
            test_size = 32153
        elif dataset_type == 'train':
            test_size = 191961
        elif dataset_type  =='test':
            test_size=60000
        else:
            raise ValueError('not valid dataset type')
            exit()

    image_dir = os.path.join(dataset_type,'image')
    # ratio to divide up the images
    #train = 0.7
    #val = 0.2
    #test = 0.1
    #if (train + test + val) != 1.0:
    #    print("probabilities must equal 1")
    #    exit()

    # get the labels
    labels = []
    imgnames = []
    annotations = {}

    with open(filename, 'r') as labelfile:
        label_string = ""
        for line in labelfile:
            labels.append(line.rstrip())

    # should be changed to whichever operation we are performing 
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            img = filename.rstrip('.jpg')
            imgnames.append(img)

    print("Labels:", labels, "imgcnt:", len(imgnames))

    # initialise annotation list
    for label in labels:
        annotations[label] = []



    print('Scanning for All Annotation Files')
    #change to len(imgnames) when ready for deployment
    # Scan the annotations for the labels
    for img in imgnames[:test_size]:
        annote = os.path.join(dataset_type,'Annotations',img+'.xml')
        if os.path.isfile(annote):
            tree = ET.parse(annote)
            root = tree.getroot()
            annote_labels = []
            for labelname in root.findall('*/name'):
                labelname = labelname.text
                annote_labels.append(labelname)
                if labelname in labels:
                    annotations[labelname].append(img)
            annotations[img] = annote_labels
        else:
            print("Missing annotation for ", annote)
            exit() 

    # create the ImageSet files

    print('Creating ImageSet Files for: %s' % dataset)
    imageset_dir = os.path.join(dataset_type,'ImageSets/Main')
    create_folder(imageset_dir)  
    with open(os.path.join(imageset_dir,dataset_type+".txt"), 'w') as outfile:
        for name in imgnames[:test_size]:
            outfile.write(name + "\n")
        for label in labels:
            with open(os.path.join(imageset_dir,label+'_'+dataset_type+".txt"), 'w') as outfile:
                for name in imgnames[:test_size]:
                    if label in annotations[name]:
                        outfile.write(name + " 1\n")
                    else:
                        outfile.write(name + " -1\n")

def create_folder(foldername):
    if os.path.exists(foldername):
        print('folder already exists:', foldername)
    else:
        os.makedirs(foldername)

if __name__=='__main__':
    if len(sys.argv) < 2 or len(sys.argv)>4:
        print("usage: python generate_deepfashion_data.py <labelfile> <dataset_type> (optional)")
    elif len(sys.argv) ==2:
        main(sys.argv[1])
    elif len(sys.argv) ==3:
        main(sys.argv[1] , sys.argv[2])
    exit()