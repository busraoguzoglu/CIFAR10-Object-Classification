import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def tsne_visualization(features, labels):

    X = features.cpu().data.numpy()
    X_rounded = np.round(X, decimals=1)

    print(X_rounded.shape)

    nsamples, nx, ny, nz = X_rounded.shape
    X_rounded = X_rounded.reshape((nsamples, nx * ny * nz))

    print(X_rounded.shape)

    print('tsne model will be created')
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(X_rounded)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    colors = ['red', 'green', 'blue', 'purple', 'yellow', 'orange', 'pink', 'magenta', 'cyan', 'black']

    plt.figure(figsize=(12, 12))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], color= colors[labels[i]-1])

    plt.show()

def tsne_visualization2(features, labels):

    X = features.cpu().data.numpy()
    X_rounded = np.round(X, decimals=1)

    print(X_rounded.shape)

    print('tsne model will be created')
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(X_rounded)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    colors = ['red', 'green', 'blue', 'purple', 'yellow', 'orange', 'pink', 'magenta', 'cyan', 'black']

    plt.figure(figsize=(12, 12))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], color= colors[labels[i]-1])

    plt.show()

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
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)

        # Pooling
        # kernel size
        self.pool1 = nn.MaxPool2d(2, 2)

        # Linear Layers
        # Batch Norm for Linear Layer
        # in_features, out_features
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        # Output layer -> # of classes = 10
        self.fc3 = nn.Linear(128, 10)

    # Define what to do on forward propagation

    # Using sigmoid vs relu activation ?
    # Using relu seem to give better results

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Shape: torch.Size([4, 6, 14, 14])
        x = self.pool1(F.relu(self.conv2(x)))  # Shape: torch.Size([4, 16, 10, 10])
        x = self.pool1(F.relu(self.conv3(x)))  # Shape: torch.Size([4, 192, 8, 8])

        # Match the input dimensions with linear layer:
        x = x.view(-1, 256 * 8 * 8)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    #def feature_extraction(self, x):
    #    x = F.relu(self.conv1(x))  # Shape: torch.Size([4, 6, 14, 14])
    #    x = self.pool1(F.relu(self.conv2(x)))  # Shape: torch.Size([4, 16, 10, 10])
    #    x = self.dropout(x)
    #    x = self.pool1(F.relu(self.conv3(x)))  # Shape: torch.Size([4, 192, 8, 8])
    #    x = self.dropout(x)

    #    return x

def calculate_accuracy(net, dataloader):
    # Get general accuracy:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_latent_space1(net, set):

    outputs = []

    with torch.no_grad():
        for data in set:
            images, labels = data
            features = net.conv1(images)
            outputs.append(features)

    return features

def get_latent_space2(net, set):

    outputs = []

    with torch.no_grad():
        for data in set:
            images, labels = data

            x = F.relu(net.conv1(images))
            x = net.pool1(F.relu(net.conv2(x)))  # Shape: torch.Size([4, 16, 10, 10])
            features = net.dropout(x)

            outputs.append(features)

    return features


def train_model(net, trainloader, trainloader2, device):
    # Define loss function
    # Option 1: Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()

    # Training Loop:
    # Number of epochs: 40
    lr = 0.001
    epochs = 40
    for epoch in range(epochs):  # loop over the dataset multiple times

        if epoch <= 20:
            optimizer = optim.Adam(net.parameters(), lr=lr)
        elif epoch > 20 and epoch <= 40:
            optimizer = optim.Adam(net.parameters(), lr=lr / 50)

        running_loss = 0.0

        # For visualizations:

        all_inputs = []
        all_labels = []

        # Loop over data
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

            # If CUDA available
            if torch.cuda.is_available():
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data

            all_inputs.append(inputs)
            all_labels.append(labels)

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

        print('here!')

        #index = 0
        #for data in trainloader2:
        #    if index == 0:
        #        # If CUDA available
        #        if torch.cuda.is_available():
        #            inputs, labels = data[0].to(device), data[1].to(device)
        #        else:
        #            inputs, labels = data
        #
        #        if epoch == 0:
        #            print('Beginning:')
        #            features = net.feature_extraction(inputs)
        #            print('Beginning to tsne:')
        #            tsne_visualization(features, labels)
        #            torch.cuda.empty_cache()
        #
        #            features = net.forward(inputs)
        #            print('Beginning to tsne2:')
        #            tsne_visualization2(features, labels)
        #            torch.cuda.empty_cache()

        #        if epoch == 20:
        #            print('Middle:')
        #            features = net.feature_extraction(inputs)
        #            print('Middle to tsne:')
        #            tsne_visualization(features, labels)
        #            torch.cuda.empty_cache()

        #            features = net.forward(inputs)
        #            print('Beginning to tsne2:')
        #            tsne_visualization2(features, labels)
        #            torch.cuda.empty_cache()

        #        if epoch == 39:
        #            print('End:')
        #            features = net.feature_extraction(inputs)
        #            print('End to tsne:')
        #            tsne_visualization(features, labels)
        #            torch.cuda.empty_cache()

        #            features = net.forward(inputs)
        #            print('Beginning to tsne2:')
        #            tsne_visualization2(features, labels)
        #            torch.cuda.empty_cache()
        #    index +=1

    print('Finished Training')
    return net

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
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

    # 2. Load the Dataset and Loaders (CIFAR10)
    trainsetnoaugmentation = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

    trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=450, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 3. Define the network
    net = Net()

    # If CUDA available
    net.to(device)

    # Define optimizer
    # Option 1: SGD with momentum
    # Option 2: Adam
    # Does not change accuracy

    net = train_model(net, trainloader, trainloader2, device)

    # Saving the trained model:
    PATH = './cifar_net2.pth'
    torch.save(net.state_dict(), PATH)

    # Test on test data:
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Working directly from the saved model:
    PATH = './cifar_net2.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # Get the highest predictions for the classes:
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    train_accuracy = calculate_accuracy(net, trainloader)
    print('Final accuracy of the network on the train images: %d %%' % (
            train_accuracy))

    test_accuracy = calculate_accuracy(net, testloader)
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            test_accuracy))

if __name__ == '__main__':
    main()