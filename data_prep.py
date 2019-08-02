import glob
import os
import csv
import h5py
import numpy as np

from PIL import Image, ImageOps
import matplotlib.pyplot as plt

'''
data_prep.py
Utility functions to preprocess and save data to low-weight formats for training
'''

#Constants
data_dir = 'adience_data'
face_dir = 'faces'
folds_to_delete = 'folds_to_delete'
minor_folds = '8_20_folds'
img_prefix = 'coarse_tilt_aligned_face'
image_size = 64

def get_all_img_path_list():
    '''
    Compile list of image paths from all pictures of subjects under 22
    '''
    img_names = []
    img_folders = '{}/{}/*'.format(data_dir, face_dir)
    for folder in img_folders:
        curr_path_head = img_folders + '/' + folder
        imgs = glob.glob('{}/*'.format(curr_path_head))
        img_names.extend(imgs)
    return img_names

def get_imgs_8_20_path_list():
    '''
    Compile list of image paths for pictures with subject
    between 8 - 20 years old (using csv files manually prepared with such info)
    '''
    img_paths = []
    fold_files = glob.glob('{}/{}/*'.format(data_dir, minor_folds))
    for fold_csv in fold_files:
        print('Opening fold csv: {}'.format(fold_csv))
        with open(fold_csv, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for img in reader:
                # print(img[2])
                if 'user_id' in img[0]:
                    print('Skipping col headers')
                    continue
                
                img_path = '{}/{}/{}/{}.*.{}'.format(data_dir, face_dir, img[0], img_prefix, img[1])
                img_name = glob.glob(img_path)
                img_paths.extend(img_name)
    return img_paths

def make_h5py_from_list(dataset, data_name):
    ''' Save a list as a h5 file '''
    with h5py.File('{}.h5'.format(data_name), 'w') as hf:
            hf.create_dataset(data_name, data=dataset)


def format_img(file_name, offset=0.0):
    ''' 
    Scale to image size (64,64)
    Normalize image between (-1,1)
    Can offset values if needed
    Save as array
    '''
    image = Image.open(file_name)
    image = ImageOps.fit(image, (image_size, image_size))
    image_arr = np.asarray(image, dtype=float)  
    image_arr = (image_arr/(255*0.5)) - 1.0
    image_arr += offset
    return image_arr

def save_to_h5py(data_name):
    ''' Compile image data into h5 '''
    paths = get_all_img_path_list()
    data_size = len(paths)
    dataset = np.zeros((data_size, image_size, image_size, 3))
    index = 0

    for path in paths:
        print('Loading image: {}'.format(path))
        dataset[index] = format_img(path, offset=0.5)
        index += 1
    
    make_h5py_from_list(dataset, data_name)
    print('Data saved to {}.h5'.format(data_name))

def get_list_from_h5py(data_name):
    ''' Load h5 data to np array (call in train.py) '''
    with h5py.File('{}.h5'.format(data_name)) as file:
        data = file[data_name]
        data = np.array(data, dtype=np.float32)
        return data

if __name__ == '__main__':

    #Prepare data
    name = 'celeb'
    #save_to_h5py(name)
    test = get_list_from_h5py(name)
    print(np.amax(test))
    print(np.amin(test))

    #View Results
    count = 10
    for i in range(count):
        plt.subplot(2, count // 2, i+1)
        plt.imshow(test[i+50])
        plt.axis('off')
    plt.tight_layout()
    plt.show()