from __future__ import print_function, division
import torch


def train(model, device, loader, optimizer):
    model.train()

    correct = 0
    total_loss = 0

    for batch_idx, sample in enumerate(loader):
        data = sample['image']
        target = sample['target']

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.to(device)

        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(output, target)

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view(-1, 1)).sum().item()

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / batch_idx
    accuracy = 100*(correct / len(loader.dataset))

    print('\tTraining set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(avg_loss, accuracy))

    return total_loss, accuracy


def test(model, device, loader):
    model.eval()

    correct = 0
    total_loss = 0

    for batch_idx, sample in enumerate(loader):
        data = sample['image']
        target = sample['target']

        data, target = data.to(device), target.to(device)
        output = model(data)
        output = output.to(device)

        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(output, target)

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view(-1, 1)).sum().item()

        total_loss += loss.item()

    avg_loss = total_loss / batch_idx
    accuracy = 100 * (correct / len(loader.dataset))

    print('\tTesting set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(avg_loss, accuracy))

    return total_loss, accuracy