import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from best_main import Net

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def tsne_visualization(features, labels):

    if torch.cuda.is_available():
        X = features.cpu().data.numpy()
        X_rounded = np.round(X, decimals=1)
    else:
        X_rounded = np.round(features, decimals=1)

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

    # 1. Define Transformations:
    # **RandomHorizontalFLip increased test accuracy by %1**
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

    # 2. Load the Dataset and Loaders (CIFAR10)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader_tsne = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)


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

    dataiter = iter(testloader_tsne)
    images, labels = dataiter.next()

    outputs = net.get_features(images)
    tsne_visualization(outputs, labels)

    train_accuracy = calculate_accuracy(net, trainloader)
    print('Final accuracy of the network on the train images: %d %%' % (
        train_accuracy))

    test_accuracy = calculate_accuracy(net, testloader)
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        test_accuracy))

if __name__ == '__main__':
    main()