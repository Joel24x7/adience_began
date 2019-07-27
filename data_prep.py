import glob
import os
import csv
import h5py
import numpy as np

from PIL import Image, ImageOps
import matplotlib.pyplot as plt

#Constants
root_dir = 'adience_data'
image_dirs = 'faces'
delete_folds = 'folds_to_delete'
prefix = 'coarse_tilt_aligned_face'
data_name='adience'
image_size = 64

#Helper Function to cleanup dataset directories
def delete_landmark_files():

    img_folders = glob.glob('{}/{}/*'.format(root_dir, image_dirs))

    for folder in img_folders:
        file_names = glob.glob('{}/*'.format(folder))
        for file in file_names:
            if 'landmarks' in file:
                os.remove(file)

def delete_old_images():

    fold_files = glob.glob('{}/{}/*'.format(root_dir, delete_folds))

    for fold_csv in fold_files:
        print('Filtering: {}'.format(fold_csv))
        with open(fold_csv, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if 'user_id' in row[0]:
                    print('Skipping col headers')
                    continue

                path = '{}/{}/{}/{}.*.{}'.format(root_dir, image_dirs, row[0], prefix, row[1])
                img_name = glob.glob(path)
                for img in img_name:
                    os.remove(img)

def delete_emptied_dirs():
    
    head = '{}/{}'.format(root_dir, image_dirs)
    for _, dir, _ in os.walk(head):
        for folder in dir:
            path = '{}/{}'.format(head, folder)
            if len(os.listdir(path)) == 0:
                os.rmdir(path)

#Helper functions for loading data
def make_h5py_from_list(dataset, data_name=data_name):
    with h5py.File('{}.h5'.format(data_name), 'w') as hf:
            hf.create_dataset(data_name, data=dataset)

def get_list_from_h5py(data_name=data_name):
    with h5py.File('{}.h5'.format(data_name)) as file:
        data = file[data_name]
        data = np.array(data, dtype=np.float32)
        return data

def get_img_path_list():

    img_names = []
    img_folders = '{}/{}/*'.format(root_dir, image_dirs)
    for folder in img_folders:
        curr_path_head = img_folders + '/' + folder
        imgs = glob.glob('{}/*'.format(curr_path_head))
        img_names.extend(imgs)
    return img_names

def format_img(file_name):
    image = Image.open(file_name)
    image = ImageOps.fit(image, (image_size, image_size))
    image_arr = np.asarray(image, dtype=float)  
    image_arr = (image_arr/(255*0.5)) - 1.0
    return image_arr

def save_to_h5py():
    paths = get_img_path_list()
    data_size = len(paths)
    dataset = np.zeros((data_size, image_size, image_size, 3))
    index = 0

    for path in paths:
        print('Loading image: {}'.format(path))
        dataset[index] = format_img(path)
        index += 1
    
    make_h5py_from_list(dataset)
    print('Data saved to {}.h5'.format(data_name))

if __name__ == '__main__':
    save_to_h5py()
    test = get_list_from_h5py()
    count = 10
    for i in range(count):
        plt.subplot(2, count // 2, i+1)
        plt.imshow(test[i + 100])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
