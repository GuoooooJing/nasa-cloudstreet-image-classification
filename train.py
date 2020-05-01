import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import csv
from ml_model.alexnet import ConvNet
from ml_model.inception_v3 import inception_v3
from ml_model.resnet import resnet18
import torchvision.models as models

NO_PATH = "./cloudstreet/no/"
YES_PATH = "./cloudstreet/yes/"
TRAIN_NO_PATH = "./dataset/train/0/"
TRAIN_YES_PATH = "./dataset/train/1/"
TEST_NO_PATH = "./dataset/test/0/"
TEST_YES_PATH = "./dataset/test/1/"
train_no_num = 300
train_yes_num = 400
test_no_num = 200
test_yes_num = 200

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
# Hyper parameters
num_epochs = 100

num_classes = 2
batch_size = 64
learning_rate = 0.01
weight_decay = 1e-3
nn_model = 'resnet'
pretrained = True
feature_extract = True


def main():
    # MNIST dataset

    train_dataset = torchvision.datasets.ImageFolder(root='./dataset/train/',
                                                     transform=transforms.ToTensor())

    test_dataset = torchvision.datasets.ImageFolder(root='./dataset/test/',
                                                    transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    print("finish loading==========================================")
    if 'loss.csv' in os.listdir('./'):
        os.remove('loss.csv')
    if 'correct.csv' in os.listdir('./'):
        os.remove('correct.csv')
    print(len(train_loader), len(test_loader))
    # model = ConvNet(num_classes).to(device)
    # model.load_state_dict(torch.load('model.ckpt'))
    model = model_init(nn_model, num_classes=num_classes, use_pretrained=pretrained, feature_extract=feature_extract)
    # model = models.inception_v3(num_classes=2)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, epoch, num_epochs)
        validate(model, test_loader, epoch)
    torch.save(model.state_dict(), 'model.ckpt')


def model_init(nn_model, num_classes = 2, use_pretrained=True, feature_extract=True):
    model = None
    if nn_model == 'resnet':
        """ Resnet18
        """
        model = resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif nn_model == 'twolayer':
        """ twolayer
        """
        model = ConvNet(num_classes)
    # elif  nn_model == 'alexnet':
    #     model = models.alexnet(pretrained=use_pretrained)
    #     set_parameter_requires_grad(model, feature_extract)
    #     num_ftrs = model.classifier[6].in_features
    #     model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    else:
        """inception_v3
        """
        model = inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)


    return model.to(device)


def train(model, train_loader, criterion, optimizer, epoch, num_epochs):
    # Train the model
    model.train()
    total_step = len(train_loader)
    loss_rate = 0
    print("start epoch # {}==========================================".format(epoch))
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        loss_rate += loss.item()
    with open(r'loss.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([loss_rate / total_step])


# Test the model
def validate(model, test_loader, epoch):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        print("start testing for epoch {}=============".format(epoch))
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)

            # print(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # print("===========================")
        # print(labels)
        # print(predicted)
        print('Test Accuracy of the model on the {} test images: {} %'.format(test_no_num + test_yes_num,
                                                                              100 * correct / total))
        with open(r'correct.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([100 * (correct / total)])


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


#

if __name__ == '__main__':
    main()
