# from operator import index
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms


train_transform = transforms.Compose([transforms.Resize(224),
                                    transforms.RandomRotation(30),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize(224, 224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder("data/train", transform=train_transform)
test_dataset = datasets.ImageFolder("data/val", transform=test_transform)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

train_iter = iter(train_loader)
test_iter = iter(test_loader)

images, labels = train_iter.next()
print(images[0], labels[0])
print('Data DUELY HANDLED')

