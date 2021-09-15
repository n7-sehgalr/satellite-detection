# from operator import index
import pandas as pd
import numpy as np
import os
import splitfolders
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

def splitting(path):
    if os.path.exists("data/dataset_splits"):
        return
    else:
        splitfolders.ratio(path, output="data/dataset_splits", seed=1337, ratio=(.8, 0.1,0.1))

splitting("data/2750")

train_transform = transforms.Compose([transforms.Resize(224),
                                    transforms.RandomRotation(30),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

val_transform = transforms.Compose([transforms.Resize(224, 224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder("data/dataset_splits/train", transform=train_transform)
val_dataset = datasets.ImageFolder("data/dataset_splits/val", transform=val_transform)
print(len(train_dataset), len(val_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

train_iter = iter(train_loader)
test_iter = iter(test_loader)

images, labels = train_iter.next()
print(images.shape, labels.shape)
print(images[0], labels[0])
print('Data DUELY HANDLED')

