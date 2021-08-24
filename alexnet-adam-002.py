import data_utils as du
import os
import time
from six.moves import urllib  # added
import sys  # added
import tarfile  # added
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import numpy as np

t1 = time.time()

np.random.seed(1000)


def maybe_download(filename, source_url, work_directory):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        print("Downloading Oxford 17 category Flower Dataset, Please "
              "wait...")
        filepath, _ = urllib.request.urlretrieve(source_url + filename,
                                                 filepath, reporthook)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')

        untar(filepath, work_directory)
        build_class_directories(os.path.join(work_directory, 'jpg'))
    return filepath

# reporthook from stackoverflow #13881092


def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))


def build_class_directories(dir):
    dir_id = 0
    class_dir = os.path.join(dir, str(dir_id))
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
    for i in range(1, 1361):
        fname = "image_" + ("%.4i" % i) + ".jpg"
        os.rename(os.path.join(dir, fname), os.path.join(class_dir, fname))
        if i % 80 == 0 and dir_id < 16:
            dir_id += 1
            class_dir = os.path.join(dir, str(dir_id))
            os.mkdir(class_dir)


def untar(fname, extract_dir):
    if fname.endswith("tar.gz") or fname.endswith("tgz"):
        tar = tarfile.open(fname)
        tar.extractall(extract_dir)
        tar.close()
        print("File Extracted")
    else:
        print("Not a tar.gz/tgz file: '%s '" % sys.argv[0])


def load_data(dirname="17flowers", resize_pics=(224, 224), shuffle=True,
              one_hot=False):
    dataset_file = os.path.join("17flowers", '17flowers.pkl')
    # if not os.path.exists(dataset_file):
    dirname = '17flowers'
    tarpath = maybe_download("17flowers.tgz",
                             "http://www.robots.ox.ac.uk/~vgg/data/flowers/17/", dirname)
    X, Y = du.build_image_dataset_from_dir(os.path.join(dirname, 'jpg/'),
                                           dataset_file=dataset_file,
                                           resize=resize_pics,
                                           filetypes=['.jpg', '.jpeg'],
                                           convert_gray=False,
                                           shuffle_data=shuffle,
                                           categorical_Y=one_hot)

    return X, Y


x, y = load_data()

# (3) Create a sequential model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11),
                 strides=(4, 4), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(
    11, 11), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(
    3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(
    3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(
    3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(17))
model.add(Activation('softmax'))

model.summary()

opt = keras.optimizers.Adam(learning_rate=0.002)

# (4) Compile
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

# (5) Correção nos shapes
y_tmp = np.zeros((x.shape[0], 17))

for i in range(0, x.shape[0]):
    y_tmp[i][y[i]] = 1
y = y_tmp

# (6) Train
model.fit(x, y, batch_size=64, epochs=100, verbose=1,
          validation_split=0.2, shuffle=True)
tempoExec = time.time() - t1
print("Tempo de execução: {} segundos".format(tempoExec))
