import os
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import csv
from model.alexnet import ConvNet
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

# Hyper parameters
num_epochs = 20
num_classes = 2
batch_size = 64
learning_rate = 0.005

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

# model = ConvNet(num_classes).to(device)
# model.load_state_dict(torch.load('model.ckpt'))
model = models.inception_v3(num_classes=2)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
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
        
        
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        loss_rate = loss.item()
    validate_rate = validate(model, test_loader)
    with open(r'loss.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([loss_rate, validate_rate])
    
    #torch.save(model.state_dict(), 'model.ckpt')

# Test the model
def validate(model, test_loader):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            print("=============")
            #print(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test images: {} %'.format(test_no_num+test_yes_num,100 * correct / total))
        return 100*(correct/total)