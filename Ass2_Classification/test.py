import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchnet import meter
import argparse
from image_loader import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument('--cuda', type=bool, default=True, help='Whether to use CUDA for training')
arg = parser.parse_args()
tada = arg.path

tdata = load_csv(csv_file='./assignment2/labels/Test_labels.csv', root_dir='./assignment2/test_classes')
isic_dataset = data_to_tensor(tdata, transform=transforms.Compose([
                              ToTensor(),
                              ]))
test_loader = torch.utils.data.DataLoader(dataset=isic_dataset, batch_size=1, shuffle=False)

#device CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.densenet169(pretrained=False)
ft = model.classifier.in_features
model.classifier = torch.nn.Linear(ft, 7)
model = model.to(device)
model.load_state_dict(torch.load('DenseNet169_retest_4'))

CM = meter.ConfusionMeter(7, normalized=True)
model.eval()
correct = 0


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

with torch.no_grad():
    preds = []
    trues = []

    for single in test_loader:
        image = single['image']
        label = single['class_label']
        images = image.to(device=device, dtype=torch.float32)
        labels = label.to(device=device, dtype=torch.float32)
        outputs = model(images)
        labels = torch.max(labels, 1)[1]
        preds.append(torch.max(outputs, 1)[1].item())
        trues.append(labels.item())
        CM.add(outputs, labels)
        correct += torch.max(outputs, 1)[1].eq(labels).sum().item()
    # Compute confusion matrix
    #cnf_matrix = confusion_matrix(labels, outputs)

    cnf_matrix = confusion_matrix(trues, preds)
    np.set_printoptions(precision=2)

    # # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, labels=["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
    # title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=np.array(["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]), normalize=True, title='Normalized confusion matrix')
    plt.show()
    accuracy = accuracy_score(trues, preds, normalize=True)
    precision = precision_score(trues, preds, average=None)
    recall = recall_score(trues, preds, average=None)
    print("Accuracy: {:.1f}%\n".format(100.*accuracy))
    fscore = 2.0 * precision * recall / (precision + recall)
    print("Recall: {}\nPrecision: {}\nFscore: {}\n".format(np.float16(recall), np.float16(precision), np.float16(fscore)))