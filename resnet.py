import os
import numpy as np
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input, Add, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.applications import resnet50
from keras.utils import to_categorical
from keras.models import Model

dataset_location = '17flowers-npz'
datafile = os.path.join(dataset_location, '17flowers.npz')
if not os.path.isfile(datafile):
    print('Run StanfordDogs.ipynb notebook first!')
    raise


npzfile = np.load(datafile)
train_images_raw = npzfile['train_images']
train_labels_raw = npzfile['train_labels']
valid_images_raw = npzfile['valid_images']
valid_labels_raw = npzfile['valid_labels']

train_images = resnet50.preprocess_input(train_images_raw.astype(np.float32))
valid_images = resnet50.preprocess_input(valid_images_raw.astype(np.float32))
print('train_images.shape:', train_images.shape)
print('valid_images.shape:', valid_images.shape)
print('valid_images:\n', valid_images[0,:,:,0].round())  # first image, red channel


train_labels = to_categorical(train_labels_raw)
valid_labels = to_categorical(valid_labels_raw)
print('train_labels.shape:', train_labels.shape)
print('valid_labels.shape:', valid_labels.shape)
print('valid_labels:\n', valid_labels)


def residual_block(X_start, filters, name, reduce=False, res_conv2d=False):
    """
    Residual building block used by ResNet-50
    """
    nb_filters_1, nb_filters_2, nb_filters_3 = filters
    strides_1 = [2,2] if reduce else [1,1]

    X = Conv2D(filters=nb_filters_1, kernel_size=[1,1], strides=strides_1, padding='same', name=name)(X_start)
    X = BatchNormalization()(X)      # default axis-1 is ok
    X = Activation('relu')(X)

    X = Conv2D(filters=nb_filters_2, kernel_size=[3,3], strides=[1,1], padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=nb_filters_3, kernel_size=[1,1], strides=[1,1], padding='same')(X)
    X = BatchNormalization()(X)

    if res_conv2d:
        X_res = Conv2D(filters=nb_filters_3, kernel_size=[1,1], strides=strides_1, padding='same')(X_start)
        X_res = BatchNormalization()(X_res)
    else:
        X_res = X_start

    X = Add()([X, X_res])
    X = Activation('relu')(X)
    return X

def create_run_model(execution_name, optimizer, input_shape, nb_classes):

    X_input = Input(shape=input_shape)

    # conv1
    X = Conv2D(filters=64, kernel_size=[7,7], strides=[2,2], padding='same', name='conv1')(X_input)
    X = BatchNormalization(name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D([3,3], strides=[2,2])(X)

    # conv2_x
    X = residual_block(X, filters=[64, 64, 256], name='conv2_a', reduce=False, res_conv2d=True)
    X = residual_block(X, filters=[64, 64, 256], name='conv2_b')
    X = residual_block(X, filters=[64, 64, 256], name='conv2_c')

    # conv3_x
    X = residual_block(X, filters=[128, 128, 512], name='conv3_a', reduce=True, res_conv2d=True)
    X = residual_block(X, filters=[128, 128, 512], name='conv3_b')
    X = residual_block(X, filters=[128, 128, 512], name='conv3_c')
    X = residual_block(X, filters=[128, 128, 512], name='conv3_d')

    # conv4_x
    X = residual_block(X, filters=[256, 256, 1024], name='conv4_a', reduce=True, res_conv2d=True)
    X = residual_block(X, filters=[256, 256, 1024], name='conv4_b')
    X = residual_block(X, filters=[256, 256, 1024], name='conv4_c')
    X = residual_block(X, filters=[256, 256, 1024], name='conv4_d')
    X = residual_block(X, filters=[256, 256, 1024], name='conv4_e')
    X = residual_block(X, filters=[256, 256, 1024], name='conv4_f')

    # conv5_x
    X = residual_block(X, filters=[512, 512, 2048], name='conv5_a', reduce=True, res_conv2d=True)
    X = residual_block(X, filters=[512, 512, 2048], name='conv5_b')
    X = residual_block(X, filters=[512, 512, 2048], name='conv5_c')

    X = GlobalAveragePooling2D(name='avg_pool')(X)
    X = Flatten()(X)
    X = Dense(units=nb_classes, activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X)

    hyps = {"OPTIMIZER_NAME": True,
    "LEARNING_RATE": True,
    "DECAY": True,
    "MOMENTUM": True,
    "NUM_EPOCHS": True,
    "BATCH_SIZE": True,
    "NUM_LAYERS": True}

    model.provenance(dataflow_tag=execution_name,
                 adaptation=True,
                 hyps = hyps)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

executions = [
    {'execution_name': 'resnet-100x-adam-0.001', 'optimizer': Adam(learning_rate=0.001)},
    {'execution_name': 'resnet-100x-adam-0.002', 'optimizer': Adam(learning_rate=0.002)},
    {'execution_name': 'resnet-100x-adam-0.0005', 'optimizer': Adam(learning_rate=0.0005)},
    {'execution_name': 'resnet-100x-sgd-0.001', 'optimizer': SGD(learning_rate=0.001)},
    {'execution_name': 'resnet-100x-sgd-0.002', 'optimizer': SGD(learning_rate=0.002)},
    {'execution_name': 'resnet-100x-sgd-0.0005', 'optimizer': SGD(learning_rate=0.0005)}
]

for execution in executions:
    np.random.seed(1000)
    create_run_model(execution['execution_name'], execution['optimizer'], input_shape=[224, 224, 3], nb_classes=17)
    