from torch import nn
import torch.nn.functional as F
from torchvision import models

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

        return x


model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
model.fc = SatImgNet()
