import zipfile
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
import matplotlib.pyplot as plt
import os
import shutil
from functions import *
# Creting model and adding filters
def create_model():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3), padding='valid', activation='relu',input_shape=(256,256,3)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

    model.add(Conv2D(64,kernel_size=(3,3), padding='valid', activation='relu',input_shape=(256,256,3)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

    model.add(Conv2D(128,kernel_size=(3,3), padding='valid', activation='relu',input_shape=(256,256,3)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    return model

    # Normalizing
def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

def main():
    # Extracting files
    zip_arch = zipfile.ZipFile('cats_and_dogs.zip','r')
    create_clear_dir("cats_and_dogs")
    zip_arch.extractall('cats_and_dogs')
    zip_arch.close()
    # Clearing data
    check_and_filter_data('cats_and_dogs/PetImages/Cat/')
    check_and_filter_data('cats_and_dogs/PetImages/Dog/')
    # Creating datasets
    train_ds,test_ds = create_sets(32, (256,256))
    
    
    # Normalizing
    train_ds = train_ds.map(process)
    test_ds = test_ds.map(process)
    # Creating model
    model = create_model()
    # Compiling model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    history = model.fit(train_ds,epochs=10,validation_data=test_ds)

if __name__=="__main__":
    main()