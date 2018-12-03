import os
import matplotlib.pyplot as plt
from skimage import io, transform
import torch
import numpy as np

# classes = pd.read_csv("./assignment2/labels/Train_labels.csv", sep=',')
# n=522
# img_name = classes.iloc[n, 0]
# #classes.head()
#
# def show_landmarks(image):
#     """Show image with landmarks"""
#     plt.imshow(image)
#     plt.pause(0.001)  # pause a bit so that plots are updated
#
# plt.figure()
# data = os.path.join('./assignment2/train/', img_name)
# img = "{}.jpg".format(img_name)
# show_landmarks(io.imread(os.path.join('.+/assignment2/train', img)))
# plt.show()


def load_csv(csv_file, root_dir):
    """
    Args:
        csv_file (string): Path to the txt file with annotations.
        root_dir (string): Directory with all the images.
    """
    with open(csv_file) as f:
        lines = f.readlines()

    data = {}
    for r in range(1, len(lines)):
        lines[r] = lines[r].rstrip('\n').split(',')
        data[lines[r][0]] = np.array([int(lines[r][1]), int(lines[r][2]), int(lines[r][3]), int(lines[r][4]), int(lines[r][5]), int(lines[r][6]), int(lines[r][7])], dtype=np.float32)

    folder_name = os.listdir(root_dir)
    dict = {}
    for folder in folder_name:
        folder_dir = root_dir + "/" + folder
        img_name = os.listdir(folder_dir)
        for img in img_name:
            simg_name = img.rstrip(".jpg").split('_')
            val = simg_name[0] + '_' + simg_name[1]
            dict[folder_dir + '/' + img] = data[val]
    return dict


# def class_list(data):
#     keys = list(data.keys())
#     classes = {}
#     for i in range(0, len(keys)):
#         cl = list(data[keys[i]])
#         c = cl.index(max(cl))
#
#         if c not in list(classes.keys()):
#             classes[c] = [i]
#         else:
#             classes[c].append(i)
#
#     classes = list(classes.values())
#     return classes

class data_to_tensor():
    """ From pytorch example"""

    def __init__(self, data, transform=None):

        self.classes = list(data.values())
        self.img = list(data.keys())
        self.transform = transform

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        img_name = self.img[idx]
        image = io.imread(img_name)
        image = image / image.max()
        image = transform.resize(image, (224, 224), mode='constant', anti_aliasing=True)
        class_label = self.classes[idx]
        sample = {'image': image, 'class_label': class_label}

        if self.transform:
           sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, class_label = sample['image'], sample['class_label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'class_label': torch.from_numpy(class_label)}

#Sanity Check
# for i in range(len(isic_dataset)):
#     sample = isic_dataset[i]
#     print(i, sample['image'].shape, sample['class_label'].shape)
#
#     if i == 5:
#         break

