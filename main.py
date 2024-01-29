import zipfile
import tensorflow as tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
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

def main():
    # Extracting files
    zip_arch = zipfile.ZipFile('cats_and_dogs.zip','r')
    if not os.path.exists("cats_and_dogs"):
        zip_arch.extractall('cats_and_dogs')
    zip_arch.close()

if __name__=="__main__":
    main()