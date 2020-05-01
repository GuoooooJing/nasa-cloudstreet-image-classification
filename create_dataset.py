import torch
from PIL import Image
import os
import torchvision.models as models

# YES: 2049 NO: 4525
NO_PATH = "./cloudstreet/no/"
YES_PATH = "./cloudstreet/yes/"
TRAIN_NO_PATH = "./new_dataset/train/0/"
TRAIN_YES_PATH = "./new_dataset/train/1/"
TEST_NO_PATH = "./new_dataset/test/0/"
TEST_YES_PATH = "./new_dataset/test/1/"

train_no_num = 1400
train_yes_num = 1400
test_no_num =  600
test_yes_num = 600

lst_no = os.listdir(NO_PATH)
lst_yes = os.listdir(YES_PATH)

import random

datalst_no = random.sample(lst_no, k=(train_no_num + test_no_num))
datalst_yes = random.sample(lst_yes, k=(train_yes_num + test_yes_num))

for i in os.listdir(TRAIN_NO_PATH):
    try:
        os.remove(TRAIN_NO_PATH + i)
    except OSError as e:
        print("Error: %s : %s" % (TRAIN_NO_PATH + i, e.strerror))

for i in os.listdir(TRAIN_YES_PATH):
    try:
        os.remove(TRAIN_YES_PATH + i)
    except OSError as e:
        print("Error: %s : %s" % (TRAIN_YES_PATH + i, e.strerror))

for i in os.listdir(TEST_NO_PATH):
    try:
        os.remove(TEST_NO_PATH + i)
    except OSError as e:
        print("Error: %s : %s" % (TEST_NO_PATH + i, e.strerror))

for i in os.listdir(TEST_YES_PATH):
    try:
        os.remove(TEST_YES_PATH + i)
    except OSError as e:
        print("Error: %s : %s" % (TEST_YES_PATH + i, e.strerror))

for i, n in enumerate(datalst_no):
    img = Image.open(NO_PATH + n).resize((800, 800))
    if i < train_no_num:
        img.save(TRAIN_NO_PATH + n)
    else:
        img.save(TEST_NO_PATH + n)

for i, n in enumerate(datalst_yes):
    img = Image.open(YES_PATH + n).resize((800, 800))
    if i < train_yes_num:
        img.save(TRAIN_YES_PATH + n)
    else:
        img.save(TEST_YES_PATH + n)

print("test_0:{}, test_1:{} \n train_:{}, train_1:{}".format(len(os.listdir(TEST_NO_PATH)),
                                                             len(os.listdir(TEST_YES_PATH)),
                                                             len(os.listdir(TRAIN_NO_PATH)),
                                                             len(os.listdir(TRAIN_YES_PATH))))
