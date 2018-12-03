
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchnet import meter
import random

from image_loader import *
import resnet as res
import alexnet as alex
import wrn


def train_batch(model, single, optimizer):
    image = single['image']
    label = single['class_label']

    images = image.to(device=device, dtype=torch.float32)
    labels = label.to(device=device, dtype=torch.float32)

    # Forward pass
    outputs = model(images)
    outputs = outputs.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    loss = criterion(outputs, torch.max(labels, 1)[1].to(device))


    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def test_batch(model, single):
    vimage = single['image']
    vlabel = single['class_label']

    vimages = vimage.to(device=device, dtype=torch.float32)
    vlabels = vlabel.to(device=device, dtype=torch.float32)

    voutputs = model(vimages)
    vlabels = torch.max(vlabels, 1)[1]
    criterion = torch.nn.CrossEntropyLoss().to(device)
    viloss = criterion(voutputs, vlabels)

    correct = torch.max(voutputs, 1)[1].eq(vlabels).sum().item()

    return correct, viloss.item(), voutputs, vlabels


#split training data
tdata = load_csv(csv_file='./assignment2/labels/Train_labels.csv', root_dir='./assignment2/train_classes')

num_train = len(tdata)
indices = list(range(num_train))
split = 5000
end = 5000
random.shuffle(indices)
train_idx, validation_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)

#device CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'pin_memory': True} if  torch.cuda.is_available() else {}

isic_dataset = data_to_tensor(tdata, transform=transforms.Compose([
                               ToTensor(),
                               ]))
train_loader = torch.utils.data.DataLoader(dataset=isic_dataset, batch_size=50, sampler=train_sampler, **kwargs)
val_loader = torch.utils.data.DataLoader(dataset=isic_dataset, batch_size=1, sampler=validation_sampler, **kwargs)

learning_rate = 0.0000001
num_epochs = 4

model = models.densenet169(pretrained=False)
ft = model.classifier.in_features
model.classifier = torch.nn.Linear(ft, 7)
model = model.to(device)
model.load_state_dict(torch.load('DenseNet169_retest_3'))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")


# Train the model
total_step = len(train_loader)


val_loss = []
train_loss = []
tloss = []
vloss = []

try:
    for epoch in range(num_epochs):
        curr_lr = learning_rate
        model.train()
        for i, single in enumerate(train_loader):
            tloss.append(train_batch(model, single, optimizer))

            if i % 10 == 0:
                print("#of images: {}/{}\n".format(i*50, len(train_loader.dataset)-5000))

        train_loss.append(sum(tloss) / len(tloss))

        confusion_matrix = meter.ConfusionMeter(7)
        model.eval()
        correct = 0
        with torch.no_grad():
            for single in val_loader:
                correct_add, viloss, voutputs, vlabels = test_batch(model, single)
                correct += correct_add
                vloss.append(viloss)
                confusion_matrix.add(voutputs, vlabels)
            val_loss.append(sum(vloss)/len(vloss))
            accur = 100.*correct / 5000
            print("Epoch [{}/{}] ---> Accuracy: {:.1f}%\n".format(epoch + 1, num_epochs, accur))
            print(confusion_matrix.conf)

        # Decay learning rate
        if (epoch + 1) % 1 == 0 & accur >= 80.00:
            curr_lr /= 10
            update_lr(optimizer, curr_lr)

        state = {'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'optim_dict' : optimizer.state_dict()}

        x = range(len(train_loss))
        plt.figure()
        plt.plot(x, train_loss, x, val_loss)
        plt.ylabel('CrossEntropy')
        plt.xlabel('# of Epochs')
        plt.legend(['Train Loss', 'Test Loss'])
        plt.show()

        # Save the model checkpoint
        torch.save(model.state_dict(), './DenseNet169_retest_4')

except:
    x = range(len(train_loss))
    plt.figure()
    plt.plot(x, train_loss, x, val_loss)
    plt.ylabel('CrossEntropy')
    plt.xlabel('# of Epochs')
    plt.legend(['Train Loss', 'Test Loss'])
    plt.show()

    # Save the model checkpoint
    torch.save(model.state_dict(), './DenseNet169_retest_4')
