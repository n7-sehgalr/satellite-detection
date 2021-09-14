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
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

test_transform = transforms.Compose([transforms.Resize(224, 224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5)) ])

train_dataset = datasets.ImageFolder("data/train", transform=train_transform)
test_dataset = datasets.ImageFolder("data/val", transform=test_transform)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

train_iter = iter(train_loader)
test_iter = iter(test_loader)

images, labels = train_iter.next()
print(images[0], labels[0])
print('Data DUELY HANDLED')

# Import Model
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(in_features=2048, out_features=10)

class SatImgNet(nn.Module):
    def __init__(self):
        super(SatImgNet, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        x = F.dropout(x)
        x = F.log_softmax(x, dim=1)


model.fc = SatImgNet()