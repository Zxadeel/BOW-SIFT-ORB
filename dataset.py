import os
import cv2
import random

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(folder+'/'+filename)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
    return images

def convert_to_bw(img_paths):
    images = []
    for filename in img_paths:
#         print(filename)
        img = cv2.imread(filename)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
    return images

def get_class_names(path):
    return os.listdir(path)

def img_list(path):
    return (os.path.join(path,f) for f in os.listdir(path))

def img_paths_list(class_names, path):
    image_paths = []
    truth = []
    prev = 0
    for i in range(len(class_names)):
        directory = path + "\\" + class_names[i]
        # print(directory)
        class_path = img_list(directory)
        image_paths += class_path
        truth += [i]*(len(image_paths)-prev)
        prev = len(image_paths)

    return image_paths, truth

def train_test_split(image_paths, truth, split_ratio=0.68):
    dataset = []
    for i in range(len(image_paths)):
        dataset.append((image_paths[i], truth[i]))
    
    random.shuffle(dataset)
    split = int(len(dataset) * split_ratio)
    # test_split = len(dataset) - train_split
    
    training = dataset[:split]
    testing = dataset[split:]
    image_paths_train, truth_train = zip(*training)
    image_paths_test, truth_test = zip(*testing)
    
    return image_paths_train, truth_train, image_paths_test, truth_test


