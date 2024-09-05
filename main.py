from os import getcwd, path

from sys import argv

from image_download import image_check
from data_check_and_generate import data_exist_or_not
from data_labeling import checking_dataset
from learning import loading_npy

ipt = argv
lst = ipt[1].split("_")
optimizer = lst[0]
learning_rate = lst[1]
loss_fn = lst[2]
epochs = lst[3]
file_number = lst[4]

home_dir = getcwd()

img_dir = path.join(home_dir, "img")

#image_check(home_dir)
data_exist_or_not(home_dir, img_dir)
checking_dataset(home_dir)

loading_npy(home_dir, optimizer, learning_rate, loss_fn, epochs, file_number)



