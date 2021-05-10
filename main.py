import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):

    # Define the layers
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional Layers
        # Batch Norm for Conv Layers
        # in_channels, out_channels, kernel_size
        self.conv1 = nn.Conv2d(3, 48, 3, padding=1)
        self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(48)
        self.conv3_bn = nn.BatchNorm2d(96)

        # Pooling
        # kernel size
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)

        # Dropout
        self.dropout = nn.Dropout(p=0.25)

        # Linear Layers
        # Batch Norm for Linear Layer
        # in_features, out_features
        self.fc1 = nn.Linear(192 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        # Output layer -> # of classes = 10
        self.fc4 = nn.Linear(64, 10)


    # Define what to do on forward propagation

    # Using sigmoid vs relu activation ?
    # Using relu seem to give better results

    def forward(self, x):
        x = F.relu(self.conv1(x))                    # Shape: torch.Size([4, 6, 14, 14])
        x = self.pool1(F.relu(self.conv2(x)))    # Shape: torch.Size([4, 16, 10, 10])
        x = self.dropout(x)
        x = self.pool1(F.relu(self.conv3(x)))        # Shape: torch.Size([4, 192, 8, 8])
        x = self.dropout(x)

        # Match the input dimensions with linear layer:
        x = x.view(-1, 192 * 8 * 8)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)

    # 1. Load and normalizing the CIFAR10 training and test datasets using torchvision
    # 2. Define a Convolutional Neural Network
    # 3. Define a loss function
    # 4. Train the network on the training data
    # 5. Test the network on the test data

    # 1. Define Transformations:
    # **RandomHorizontalFLip increased test accuracy by %1**
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 2. Load the Dataset and Loaders (CIFAR10)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 3. Define the network
    net = Net()

    # If CUDA available
    net.to(device)

    # Define loss function
    # Option 1: Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    # Option 1: SGD with momentum
    # Option 2: Adam
    # Does not change accuracy

    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Training Loop:
    # Number of epochs: 5
    for epoch in range(14):  # loop over the dataset multiple times

        running_loss = 0.0

        # Loop over data
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

            #inputs, labels = data

            # If CUDA available
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # backward propagation and weight update
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # Saving the trained model:
    PATH = './cifar_net3.pth'
    torch.save(net.state_dict(), PATH)

    # Test on test data:
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Working directly from the saved model:
    net = Net()
    net.load_state_dict(torch.load(PATH))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # Get the highest predictions for the classes:
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    # Get general accuracy:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

if __name__ == '__main__':
    main()