from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader
from torchvision import models
from data import *
from model import *
from utils import *
import random

def test_model(null_split):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_data = train_files('/home/hansencb/SkinLesions/Test')

    test_dataset = Test_Dataset(test_data)

    batch_size = 16
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)


    model = models.densenet169(pretrained=True)
    ft = model.classifier.in_features
    model.classifier = torch.nn.Linear(ft, 7)
    model = model.to(device)

    model_file = 'models/saved_model_split_{}'.format(null_split)
    model.load_state_dict(torch.load(model_file))

    loss, accuracy = test(model, device, test_loader)
    return accuracy


def main():
    #splits = [1000, 2000, 3000, 4000, 5000, 5500, 6000, 6500, 7000, 7500, 7750, 8000]
    splits = [8000]

    test_acc_file = 'results/test_accuracy.txt'
    f = open(test_acc_file, 'w')
		
    for split in splits:
        print('Testing model with split {}'.format(split))
        accuracy = test_model(split)
        f.write(str(accuracy)+'\n')
    f.close()
		

if __name__ == '__main__':
    main()
