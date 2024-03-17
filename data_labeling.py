import numpy as np

from os import path, mkdir, listdir, remove

from tqdm import tqdm
from PIL import Image

def append_to_list(dir_name, item, train_images, train_labels, test_images, test_labels, image_or_label):
    if dir_name == "train":
        if image_or_label == "image":
            train_images.append(item)
        elif image_or_label == "label":
            train_labels.append(item)

    elif dir_name == "test":
        if image_or_label == "image":
            test_images.append(item)
        elif image_or_label == "label":
            test_labels.append(item)

def checking_dataset(home_dir):
    dataset_dir = path.join(home_dir, "dataset")
    files = ["train_image.npy", "train_labels.npy", "test_image.npy", "test_labels.npy"]

    if not ("dataset" in listdir(home_dir)):
        mkdir("dataset")
        labeling(home_dir, dataset_dir)

    elif not (file in listdir(dataset_dir) for file in files):
        for i in listdir(dataset_dir):
            remove(path.join(dataset_dir, i))
        labeling(home_dir, dataset_dir)

    else:
        print("Dataset generating is done")


# noinspection PyUnboundLocalVariable
def labeling(home_dir, dataset_dir):
    data_dir = path.join(home_dir, 'data')

    red, yellow, green = ["빨간신호", "빨간불"], ["노랑신호", "노란불"], ["초록신호", "초록불"]

    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for dir_name in ["train", "test"]:
        directory = path.join(data_dir, dir_name)

        for item in tqdm(listdir(directory)):
            location = path.join(directory, item)
            img = Image.open(location).resize((255, 255))
            img = np.array(img)

            if img.shape[2] == 3:
                alpha_channel = np.zeros((255, 255, 1))
                img_2 = np.concatenate([img, alpha_channel], axis = 2)

                append_to_list(dir_name, img_2, train_images, train_labels, test_images, test_labels, "image")

            else:
                append_to_list(dir_name, img, train_images, train_labels, test_images, test_labels, "image")

            for i in range(2):
                if red[i] in item:
                    append_to_list(dir_name, 0, train_images, train_labels, test_images, test_labels, "label")
                    break

                elif yellow[i] in item:
                    append_to_list(dir_name, 1, train_images, train_labels, test_images, test_labels, "label")
                    break

                elif green[i] in item:
                    append_to_list(dir_name, 2, train_images, train_labels, test_images, test_labels, "label")
                    break

    if len(train_images) != len(train_labels):
        print("error in trains")
        exit(1)

    elif len(test_images) != len(test_labels):
        print("error in test")
        exit(1)

    train_images, train_labels = np.array(train_images), np.array(train_labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)

    np.save(path.join(dataset_dir, "train_image.npy"), train_images)
    print("train_image.npy is generated")

    np.save(path.join(dataset_dir, "train_labels.npy"), train_labels)
    print("train_labels.npy is generated")

    np.save(path.join(dataset_dir, "test_image.npy"), test_images)
    print("test_image.npy is generated")

    np.save(path.join(dataset_dir, "test_labels.npy"), test_labels)
    print("test_labels.npy is generated")

    print("Data labeling and generating dataset are done")
