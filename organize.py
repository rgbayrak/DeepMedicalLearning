import pickle
import os
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


labels_file = '/home/hansencb/SkinLesions/labels/Train_labels.csv'
folder = '/home/hansencb/SkinLesions/Train'
with open(labels_file, 'r') as f:
    lines = f.readlines()
    for i in range(1,len(lines)):
        info = lines[i].strip().split(',')
        filename = os.path.join(folder, info[0]+'.jpg')

        label = np.argmax(list(map(int, info[1:])))
        new_name = os.path.join(folder, str(label), info[0]+'.jpg')
        os.rename(filename, new_name)


