import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Net(nn.Module):
    # Define the layers
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional Layers
        # Batch Norm for Conv Layers
        # in_channels, out_channels, kernel_size
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)

        # Pooling
        self.pool1 = nn.MaxPool2d(2, 2)

        # Dropout
        self.dropout = nn.Dropout(p=0.20)

        # Linear Layers
        # Batch Norm for Linear Layer
        # in_features, out_features
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dense2_bn = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 32)
        # Output layer -> # of classes = 10
        self.fc4 = nn.Linear(32, 10)

    # Define what to do on forward propagation
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(self.conv2_bn(self.conv2(x)))
        x = F.relu(x)
        x = self.pool1(self.conv3(x))
        x = F.relu(x)
        x = self.dropout(x)

        # Match the input dimensions with linear layer:
        x = x.view(-1, 128 * 8 * 8)

        x = F.relu(self.fc1(x))
        x = F.relu(self.dense2_bn(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_model(net, trainloader, device):

    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    epochs = 40
    all_loss = []

    # Training Loop:
    for epoch in range(epochs):  # loop over the dataset multiple times

        if epoch <= 20:
            optimizer = optim.Adam(net.parameters(), lr=lr)
        elif epoch > 30 and epoch <= 80:
            optimizer = optim.Adam(net.parameters(), lr=lr / 50)

        running_loss = 0.0

        # Loop over data
        for i, data in enumerate(trainloader, 0):
            # If CUDA available
            if torch.cuda.is_available():
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data

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
                all_loss.append(running_loss / 2000)
                running_loss = 0.0

    # show loss curve
    plt.plot(all_loss)
    plt.show()

    print('Finished Training')
    return net