import zipfile
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
import matplotlib.pyplot as plt
import os
import shutil

# Creating directory or clearing it, if already exists
def create_clear_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        file_list = os.listdir(path)
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    print(f"{file_name} is not a file.")
            except Exception as e:
                print(f"Error while removing file: {file_name}: {e}")

# Dividing files in different sets
def make_a_set(path,dog_folder,cat_folder,which_half):
    # Create subfolders for cat and dog
    train_cat_folder = os.path.join(path, "cat")
    train_dog_folder = os.path.join(path, "dog")
    create_clear_dir(train_cat_folder)
    create_clear_dir(train_dog_folder)
    # Upload cat and dog photos
    cat_files = os.listdir(cat_folder)
    dog_files = os.listdir(dog_folder)
    if which_half == 1:
        half_cat_files = cat_files[:len(cat_files)//2]
        half_dog_files = dog_files[:len(dog_files)//2]
    elif which_half == 2:
        half_cat_files = cat_files[len(cat_files)//2:]
        half_dog_files = dog_files[len(dog_files)//2:]
    else:
        raise Exception("Wrong half number in function make_a_set")
    
    for file_name in half_cat_files:
        source_path = os.path.join(cat_folder, file_name)
        dest_path = os.path.join(train_cat_folder, file_name)
        shutil.move(source_path, dest_path)

    for file_name in half_dog_files:
        source_path = os.path.join(dog_folder, file_name)
        dest_path = os.path.join(train_dog_folder, file_name)
        shutil.move(source_path, dest_path)

# Dividing data to test and train sets
def create_sets(batch_size, image_size):
    cat_folder = 'cats_and_dogs/PetImages/Cat/'
    dog_folder = 'cats_and_dogs/PetImages/Dog/'
    train_folder = 'train'
    test_folder = 'test'
    # Creating/clearing train folder 
    create_clear_dir(train_folder)
    # Uploading data to train folder 
    make_a_set(train_folder,dog_folder,cat_folder,1)
    # Creating/clearing test folder 
    create_clear_dir(test_folder)
    # Uploading data to test folder
    make_a_set(test_folder,dog_folder,cat_folder,2)
    # Creating keras sets
        # Train
    train_ds = keras.utils.image_dataset_from_directory(
    directory = 'train' ,
    labels = 'inferred',
    label_mode = 'int',
    batch_size = batch_size,
    image_size = image_size)
        # Test
    test_ds = keras.utils.image_dataset_from_directory(
    directory = 'test' ,
    labels = 'inferred',
    label_mode = 'int',
    batch_size = batch_size,
    image_size = image_size)
    return train_ds,test_ds

# Checking if image is broken
def is_image_valid(image_path):
    try:
        # Load the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image)      
        # Check channel number
        num_channels = image.shape[-1]
        return num_channels in [1, 3, 4]
    
    except tf.errors.InvalidArgumentError as e:
        print(f"Error for image: {image_path}: {e}")
        return False
    
# Checking data folder for faulty images 
def filter_images(directory):
    valid_images = []
    invalid_images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory, filename)
            if is_image_valid(image_path):
                valid_images.append(image_path)
            else:
                invalid_images.append(image_path) 
    return valid_images, invalid_images

# Deleting broken images
def check_and_filter_data(image_directory):
    valid_images, invalid_images = filter_images(image_directory)
# Delete faulty images
    for invalid_image in invalid_images:
        os.remove(invalid_image)

