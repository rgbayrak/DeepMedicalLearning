from __future__ import print_function, division
import torch
import torchvision

import os
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def train_files(folder):
    class_folders = os.listdir(folder)
    data = {}
    for c in class_folders:
        files = os.listdir(os.path.join(folder, c))

        for f in files:
            data[os.path.join(folder,c,f)] = int(c)

    return data

# def test_files(folder):
#     files = os.listdir(folder)
#     data = []
#     for f in files:
#         data.append(f)
#
#     return data

class Train_Dataset(Dataset):
    def __init__(self, data, null_split=0):
        data = data
        keys = list(data.keys())

        self.tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transforms = transforms.Compose([
            transforms.RandomAffine(10, translate=(0.1, 0.1), shear=2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter()
        ])

        self.class_to_files = {}
        self.null_class_to_files = {}
        for key in keys:
            label = data[key]
            if label not in self.class_to_files:
                self.class_to_files[label] = [key]
            else:
                self.class_to_files[label].append(key)

        class_splits = []
        for key in range(len(self.class_to_files)):
            self.class_to_files[key] = sorted(self.class_to_files[key])
            class_splits.append(int(null_split*(len(self.class_to_files[key])/len(keys))))

        if null_split > 0:
            diff = null_split - sum(class_splits)
            class_splits[np.argmax(class_splits)] += diff
            for i in range(len(class_splits)):
                class_split = class_splits[i]
                keys = self.class_to_files[i]
                split_keys = keys[0:class_split]


                self.null_class_to_files[i] = split_keys
                self.class_to_files[i] = keys[class_split:]

        self.len = 0
        for i in range(len(self.class_to_files)):
            self.len = max(self.len, len(self.class_to_files[i]))
            self.len = max(self.len, len(self.null_class_to_files[i]))

        self.len *= i


    def __len__(self):
        return self.len

    def preprocess(self, fname):
        img = io.imread(fname)

        img = Image.fromarray(img.astype('uint8'))

        if self.transforms:
            img = self.transforms(img)

        img = np.array(img)
        img = img / 255.0
        img = transform.resize(img, [224, 224, 3], mode='constant', anti_aliasing=True)
        img = self.tensor(img)
        img = self.norm(img)
        img = img.type(torch.float32)

        return img

    def __getitem__(self, idx):

        label = random.randint(0, len(self.class_to_files)-1)
        idx = random.randint(0, len(self.class_to_files[label])-1)
        fname = self.class_to_files[label][idx]

        img = self.preprocess(fname)

        null_label = random.randint(0, len(self.class_to_files)-1)
        null_keys = self.class_to_files[null_label].copy()
        idx = random.randint(0, len(null_keys) -1)
        null_image1 = self.preprocess(null_keys[idx])
        del null_keys[idx]
        idx = random.randint(0, len(null_keys) -1)
        null_image2 = self.preprocess(null_keys[idx])

        return {'image': img, \
                'target': torch.tensor(label), \
                'null_img1': null_image1,\
                'null_img2': null_image2 }



class Test_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.keys = list(data.keys())
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tensor = transforms.ToTensor()


    def __len__(self):
        return len(self.keys)


    def preprocess(self, fname):
        img = io.imread(fname)
        img = img / 255.0
        img = transform.resize(img, [224, 224, 3], mode='constant', anti_aliasing=True)

        img = self.tensor(img)
        img = self.norm(img)
        img = img.type(torch.float32)

        return img


    def __getitem__(self, idx):
        fname = self.keys[idx]
        label = self.data[fname]

        img = self.preprocess(fname)

        return {'image': img, \
                'target': torch.tensor(label)}
