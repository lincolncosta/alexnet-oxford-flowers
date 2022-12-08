import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import PIL

def download(url,dest,md5sum):
    import os
    import urllib
    import hashlib
    import urllib.request

    folder, file = os.path.split(dest)
    if folder != '':
        os.makedirs(folder, exist_ok=True)
    if not os.path.isfile(dest):
        print('Downloading', file, '...')
        urllib.request.urlretrieve(url, dest)
    else:
        print('Already Exists:', file)
    assert hashlib.md5(open(dest, 'rb').read()).hexdigest() == md5sum

def extract(src, dest):
    import os
    import tarfile

    path, file = os.path.split(src)
    extract_path, _ = os.path.splitext(src)
    already_extracted = os.path.isdir(dest)
    if not already_extracted:
        with tarfile.open(src, 'r') as zf:
            print('Extracting', file, '...')
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(zf, dest)
    else:
        print('Already Extracted:', file)
    assert os.path.isdir(extract_path)

dataset_location = './vgg-flowers-17/'

download(url='http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz',
         dest=os.path.join(dataset_location, '17flowers.tgz'),
         md5sum='b59a65d8d1a99cd66944d474e1289eab')

extract(src=os.path.join(dataset_location, '17flowers.tgz'),
        dest=os.path.join(dataset_location, '17flowers'))

all_files = np.loadtxt(os.path.join(dataset_location, '17flowers/jpg/files.txt'), dtype=str)
print('all_files.shape:', all_files.shape)
print('all_files:      ', all_files[:3])

all_labels = []
for i in range(17):
    all_labels.extend([i]*80)
all_labels = np.array(all_labels)
print('all_labels.shape:', all_labels.shape)
print('all_labels.min():', all_labels.min())
print('all_labels.max():', all_labels.max())
print('all_labels:      ', all_labels)


def load_images(folder, files, target_size):
    images_list = []
    for file in files:
        img_full_path = os.path.join(folder, file)
        img = PIL.Image.open(img_full_path)
        img = img.resize(target_size)
        images_list.append(np.array(img))
    return np.array(images_list)


all_images = load_images(folder=os.path.join(dataset_location, '17flowers/jpg'),
                         files=all_files, target_size=(224,224))



def show_images(start_index, images_array, labels_array):
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=[16,9])
    for i, ax in enumerate(axes):
        ax.imshow(images_array[start_index+i])
        ax.set_title('Idx: '+str(start_index+i)+' Label: '+str(labels_array[start_index+i]))


show_images(360, all_images, all_labels)


train_indices = []
valid_indices = []
for i in range(0, 1360, 80):
    train_indices.extend(range(i, i+70))
    valid_indices.extend(range(i+70, i+80))
train_indices = np.array(train_indices)
valid_indices = np.array(valid_indices)
# Sanity check
all_indices = sorted(np.concatenate([train_indices, valid_indices]))
assert np.alltrue(np.array(range(1360)) == all_indices)

train_labels = all_labels[train_indices]
train_images = all_images[train_indices]
valid_labels = all_labels[valid_indices]
valid_images = all_images[valid_indices]

show_images(66, train_images, train_labels)

save_path = os.path.join(dataset_location, '17flowers.npz')
print(save_path)

np.savez(save_path,
         train_images=train_images,
         train_labels=train_labels,
         valid_images=valid_images,
         valid_labels=valid_labels)