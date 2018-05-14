from keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
import os.path
import numpy as np
import os
import json
import keras.models
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from glob import glob
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam , SGD
from keras.applications.vgg16 import VGG16
from PIL import Image
from keras import regularizers
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.applications.inception_v3 import InceptionV3

size_image = 96
data = {}
data['data'] = []

paths ='food/train'

def getSum(path):
    sum = 0
    for d in os.listdir(path):
        num = len(os.listdir(os.path.join(path, d)))
        sum += num
    return sum
sum = getSum(paths)
Y_all = np.zeros(sum)
X_all = np.zeros((sum, size_image, size_image, 3), dtype='float64')

def init(path):

    count_X = 0
    label = 0
    for d in os.listdir(path):
        if os.path.isdir(os.path.join(path, d)):
            for img in os.listdir(os.path.join(path, d)):
                if img.endswith("jpg"):
                    image = Image.open(os.path.join(os.path.join(path, d), img))
                    image = image.resize((size_image, size_image), Image.ANTIALIAS)
                    image = np.array(image)
                    X_all[count_X]= image
                    Y_all[count_X] = label
                    count_X +=1

            data['data'].append({
                'id': label,
                'name': d
            })
        label +=1
    with open('data1.txt', 'w') as outfile:
        json.dump(data, outfile)


def models():
    model = Sequential()
    model.add(Convolution2D(32, (3, 3),activation='relu',input_shape=(size_image, size_image,3)))
    model.add(Convolution2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Convolution2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    

    model.add(Convolution2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01),
                    ))
    model.add(Dropout(0.5))


    model.add(Dense(sum_laber, activation='softmax'))

    return model


    

#
if __name__ == '__main__':
    init(paths)
    sum_laber = len(os.listdir(paths))

    Y_all = np_utils.to_categorical(Y_all, sum_laber)
    X_all /= 255.0

    X_train,X_test,Y_train,Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.125,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.125,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # rescale=1. / 255,
        fill_mode='nearest')
    datagen.fit(X_train)
  
    model=models()

    # #
    # model = VGG16(include_top=False,
    #               weights=None, input_tensor=Input(shape=(size_image, size_image, 3)))
    model.compile(optimizer=SGD(lr=0.01,momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='models.h5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('train1.csv')
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                        steps_per_epoch=X_train.shape[0]/32,
                        validation_data=datagen.flow(X_test,Y_test, batch_size=32),
                        epochs=50,
                        callbacks=[csv_logger,checkpointer],
                        validation_steps=Y_train.shape[0]/32,
                        )

    score = model.evaluate(X_test, Y_test, verbose=1)

   










