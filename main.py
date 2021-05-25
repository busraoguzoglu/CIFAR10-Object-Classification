import torch
import torchvision
import torchvision.transforms as transforms

from model import Net
from model import train_model

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 1. Load and normalizing the CIFAR10 training and test datasets using torchvision
    # 2. Define a Convolutional Neural Network
    # 3. Train the network on the training data
    # 4. Test the network on the test data (eval.py)

    # Define Transformations:
    # **RandomHorizontalFLip increases test accuracy by %1**
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

    # 1. Load the Dataset and Loaders for training (CIFAR10)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

    # 2. Define the network
    net = Net()

    # If CUDA available
    net.to(device)

    # 3. Train the defined network
    net = train_model(net, trainloader, device)

    # Saving the trained model:
    PATH = './cifar_net_test.pth'
    torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    main()