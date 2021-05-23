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

from best_main import Net

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

def calculate_accuracy(net, dataloader):
    net.eval()
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


def main():

    print('This is for latent space evaluation')

    # 1. Define Transformations:
    # **RandomHorizontalFLip increased test accuracy by %1**
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

    # 2. Load the Dataset and Loaders (CIFAR10)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 3. Define the network
    net = Net()

    # If CUDA available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Working directly from the saved model:
    PATH = './cifar_net8.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    outputs = net(images)

    tsne_visualization2(outputs, labels)

if __name__ == '__main__':
    main()