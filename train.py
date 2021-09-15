import numpy as np
import torch 
from torch import nn
from torch import optim
from torch.autograd import Variable
from data_handler import dataset_loader
from models import model
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary

train_path = "data/dataset_splits/train"
val_path = "data/dataset_splits/val"
test_path = "data/dataset_splits/test"

train_loader, val_loader, _ = dataset_loader(train_path, val_path, test_path)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.01)

epochs = 3

epoch_values = []
loss_values = []
valid_loss_values = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(summary(model, (3, 224, 224)))


for e in tqdm(range(epochs)):
    print(f"Epoch: {e+1}")
    running_loss = 0.0
    running_valid_loss = 0.0

    try: 
        for i, (inputs, labels) in tqdm(enumerate(iter(train_loader))):
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' % (e + 1, i + 1, running_valid_loss / 1000))

        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(iter(val_loader)):
                output = model(inputs)
                valid_loss = criterion(output, labels)
                running_valid_loss += valid_loss.item()

                if i % 100 == 99:    # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' % (e + 1, i + 1, running_valid_loss / 1000))

        epoch_values.append(e+1)    
        loss_values.append(running_loss/2000)
        valid_loss_values.append(running_valid_loss/1000)

    except Exception as e:
        with open('log.txt', 'a') as file:
            file.write(str(e))

print(summary(model, (3, 224, 224)))

plt.plot(epoch_values, loss_values)
plt.plot(epoch_values, valid_loss_values)
# plt.legend('')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.xlim((1, epochs))
plt.show()
