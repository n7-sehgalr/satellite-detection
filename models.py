import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import os

train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.RandomRotation(30),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])


test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])


base_path = "dataset/"

train_data = datasets.ImageFolder(os.path.join(base_path, "train"), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(base_path, "test"), transform=test_transform)

trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader = DataLoader(test_data, batch_size=32, shuffle=True)

# Import Models
resnet_model = models.resnet50(pretrained=True)
googlenet_model = models.googlenet(pretrained=True)

