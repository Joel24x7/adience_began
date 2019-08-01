import os
import csv
import glob

'''
Misc.py 
Utility functions to cleanup file structure of default adience dataset
'''

data_dir = 'adience_data'
face_dir = 'faces'
folds_to_delete = 'folds_to_delete'
minor_folds = '8_20_folds'
img_prefix = 'coarse_tilt_aligned_face'


def delete_landmark_files():
    '''
    Adience dataset included txt files with landmarks for facial recognition classifiers
    Delete these files to save storage
    '''
    img_folders = glob.glob('{}/{}/*'.format(data_dir, face_dir))

    for folder in img_folders:
        file_names = glob.glob('{}/*'.format(folder))
        for file in file_names:
            if 'landmarks' in file:
                os.remove(file)

def delete_old_images():
    '''
    Manually filtered csv fold files to include img paths of subjects older than 22
    Delete all images of older subjects
    '''
    fold_files = glob.glob('{}/{}/*'.format(data_dir, folds_to_delete))

    for fold_csv in fold_files:
        print('Filtering: {}'.format(fold_csv))
        with open(fold_csv, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if 'user_id' in row[0]:
                    print('Skipping col headers')
                    continue

                path = '{}/{}/{}/{}.*.{}'.format(data_dir, face_dir, row[0], img_prefix, row[1])
                img_name = glob.glob(path)
                for img in img_name:
                    os.remove(img)

def delete_emptied_dirs():
    ''' Delete directories with no images '''
    head = '{}/{}'.format(data_dir, face_dir)
    for _, dir, _ in os.walk(head):
        for folder in dir:
            path = '{}/{}'.format(head, folder)
            if len(os.listdir(path)) == 0:
                os.rmdir(path)