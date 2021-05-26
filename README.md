# CIFAR10-Object-Classification

Simple 3 layer CNN for object classification task on CIFAR10 dataset

Restriction: Using 3 Convolutional Layers

----------------------------------------------------------------------

With best model settings:

Achieved Max accuracy of the network on the train images: 94%

Achieved Max Test Accuracy: 87% (cifar_net8.pth)

Training takes around 135 minutes for 80 epochs using GPU.

----------------------------------------------------------------------------------------------------

model.py

This file includes the network, parameters regarding to network can be changed from here.

It includes the train_model function, it is called from the main.py file.

Number of epochs and learning rate was set inside the train function, it can be changed from there.

After the training finished, it draws the loss function for the specific training.

It also includes get_features function, which is different than forward function.

This function is written to execute forward function until a certain point.

----------------------------------------------------------------------------------------------------

main.py

When this file is running:

In this file, the train set is loaded at first, and transformations were defined.

Network is defined (from model.py)

train_model function is called (from model.py)

model is saved to the given path.

----------------------------------------------------------------------------------------------------

eval.py

When this file is running:

In this file, both train and test set is loaded, transformations were defined.

Trained model is loaded from the savefile, given path.

tsne visualization is done, using the get_features function from model.py

tsne visualization is currently done using 1000 test images.

train accuracy is calculated and printed for the given model.

test accuracy is calculated and printed for the given model.

