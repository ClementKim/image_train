import cv2
import imutils
import torch

import numpy as np
import matplotlib.pyplot as plt

from os import remove, path, mkdir, listdir

from PIL import Image

from random import randrange

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

def pil_to_tensor(pil_image):
    return torch.as_tensor(np.asarray(pil_image).copy()).permute(2, 0, 1)



def tensor_to_pil(tensor_image):
    return to_pil_image(tensor_image)


def tensor_to_pilimg(tensor_image):
    return tensor_image.permute(1, 2, 0).numpy()


def flip_image(original, data_dir):
    transform = transforms.RandomVerticalFlip(1)
    tensor = pil_to_tensor(original)
    flipped = transform(tensor)
    plt.imsave(data_dir, tensor_to_pilimg(flipped))


def gray_image(original, data_dir):
    transform = transforms.RandomGrayscale(1)
    tensor = pil_to_tensor(original)
    gray = transform(tensor)
    plt.imsave(data_dir, tensor_to_pilimg(gray))


def invert_image(original, data_dir):
    transform = transforms.RandomInvert(1)
    tensor = pil_to_tensor(original)
    inverted = transform(tensor)
    plt.imsave(data_dir, tensor_to_pilimg(inverted))

def clear_dir(dir):
    for item in listdir(dir):
        remove(path.join(dir, item))

def data_exist_or_not(home_dir, img_dir):
    data_dir = path.join(home_dir, 'data')
    train_dir = path.join(data_dir, 'train')
    test_dir = path.join(data_dir, 'test')

    if not ('data' in listdir(home_dir)):
        mkdir(data_dir)
        mkdir(train_dir)
        mkdir(test_dir)
        duplicate_check(img_dir, train_dir, test_dir)

    else:
        if not ('train' in listdir(data_dir)):
            mkdir(train_dir)

        if not ('test' in listdir(data_dir)):
            mkdir(test_dir)

        if not (len(listdir(img_dir)) * 36 * (15 + 15 + 3) == (len(listdir(train_dir)) + len(listdir(test_dir)))):
            duplicate_check(img_dir, train_dir, test_dir)

        print("Data is already exist")
        
def duplicate_check(img_dir, train_dir, test_dir):
    img_list = listdir(img_dir)

    arr = []
    arr_name = []
    img_name_dic = {}
    multiple_check = []
    flag = False
    for img_name in img_list:
        if img_name == ".DS_Store":
            continue

        img_path = path.join(img_dir, img_name)
        image = cv2.imread(img_path)

        try:
            # image.shape = (height, width, channel)
            # image.shape[2] == 3 => rgb
            if image.shape[2] == 3:
                if not multiple_check:
                    pass
                else:
                    for i in multiple_check:
                        if np.array_equal(i, image):
                            flag = True
                            remove(img_path)
                            break

                if not flag:
                    resized_image = cv2.resize(image, (60, 60))
                    arr.append(resized_image)
                    multiple_check.append(image)

                    try:
                        img_name_dic[arr_name[arr_name.index(img_name[:-7])]] = 1

                    except:
                        img_name_dic[img_name[:-7]] = 1

                    arr_name.append(img_name[:-7])

                else:
                    flag = False
        except:
            pass

    data = np.array(arr)

    if (len(data) * 36 * (15 + 15 + 3)) == (len(listdir(train_dir)) + len(listdir(test_dir))):
        print("Nothing to do for generating")
        return

    data_generate(train_dir, test_dir, data, img_name_dic, arr_name)

def data_generate(train_dir, test_dir, data, img_name_dic, arr_name):
    if len(listdir(train_dir)):
        clear_dir(train_dir)

    if len(listdir(test_dir)):
        clear_dir(test_dir)

    num = 0
    limit = round(36 * (15 + 15 + 3) * 0.7) - 1

    for idx in range(len(data)):
        rotate = data[idx]
        if img_name_dic[arr_name[idx]] > limit:
            generate_location = train_dir
        else:
            generate_location = test_dir

        for angle_time in range(36):
            angle = randrange(0, 360, 10)

            rotate = imutils.rotate(rotate, angle)

            img_name_dic[arr_name[idx]] += 1
            location = path.join(generate_location, arr_name[idx] + " " + str(img_name_dic[arr_name[idx]]) + ".png")

            cv2.imwrite(location, rotate)
            num += 1

            original = Image.open(location)

            transform = transforms.RandomCrop((35, 35))

            print(f"{round(num / (len(data) * 36 * (15 + 15 + 3)) * 100, 2)}%")

            for _ in range(15):
                img_name_dic[arr_name[idx]] += 1

                tensor = pil_to_tensor(original)

                cutted = transform(tensor)
                cutted_location = path.join(generate_location, arr_name[idx] + " " +
                                            str(img_name_dic[arr_name[idx]]) + ".png")

                plt.imsave(cutted_location, tensor_to_pilimg(cutted))
                num += 1

                print(f"{round(num / (len(data) * 36 * (15 + 15 + 3)) * 100, 2)}%")

            img_name_dic[arr_name[idx]] += 1
            flip_image_location = path.join(generate_location, arr_name[idx] + " " +
                                            str(img_name_dic[arr_name[idx]]) + ".png")
            flip_image(original, flip_image_location)
            num += 1
            print(f"{round(num / (len(data) * 36 * (15 + 15 + 3)) * 100, 2)}%")

            # img_name_dic[arr_name[idx]] += 1
            # gray_image_location = path.join(generate_location, arr_name[idx] + " " +
            #                                 str(img_name_dic[arr_name[idx]]) + ".png")
            # gray_image(original, gray_image_location)
            # num += 1
            # print(f"{round(num / (len(data) * 36 * (15 + 15 + 3)) * 100, 2)}%")

            for _ in range(15):
                img_name_dic[arr_name[idx]] += 1

                tensor = pil_to_tensor(original)

                transform = transforms.RandomPerspective(fill=[randrange(60), randrange(60), randrange(60)])

                randPers = transform(tensor)
                randPers_location = path.join(generate_location, arr_name[idx] + " " +
                                              str(img_name_dic[arr_name[idx]]) + ".png")

                plt.imsave(randPers_location, tensor_to_pilimg(randPers))

                num += 1
                print(f"{round(num / (len(data) * 36 * (15 + 15 + 3)) * 100, 2)}%")

            img_name_dic[arr_name[idx]] += 1
            invert_image_location = path.join(generate_location, arr_name[idx] + " " +
                                              str(img_name_dic[arr_name[idx]]) + ".png")
            invert_image(original, invert_image_location)
            num += 1
            print(f"{round(num / (len(data) * 36 * (15 + 15 + 3)) * 100, 2)}%")

            # for _ in range(15):
            #     img_name_dic[arr_name[idx]] += 1
            #
            #     tensor = pil_to_tensor(original)
            #
            #     transform = transforms.RandomSolarize(randrange(255))
            #
            #     solarized_img = transform(tensor)
            #     solarized_img_location = path.join(generate_location, arr_name[idx] + " " +
            #                                        str(img_name_dic[arr_name[idx]]) + ".png")
            #
            #     plt.imsave(solarized_img_location, tensor_to_pilimg(solarized_img))
            #
            #     num += 1
            #     print(f"{round(num / (len(data) * 36 * (15 + 15 + 3)) * 100, 2)}%")

    if len(listdir(train_dir)) + len(listdir(test_dir)) == num:
        print("Data generating is done")

    else:
        print("Error")
        exit(2)
