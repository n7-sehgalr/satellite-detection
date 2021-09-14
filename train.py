import numpy as np

from torch import nn
from torch import optim
from torch.autograd import Variable
from data_handler import trainloader
from models import model

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 3

epoch_values = []
loss_values = []
valid_loss_values = []

for e in range(epochs):
    print(f"Epoch: {e+1}")
    running_loss = 0.0
    running_valid_loss = 0.0

    for i, (inputs, labels) in enumerate(iter(trainloader)):
        output = model(inputs)
        optimizer.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (e + 1, i + 1, running_valid_loss / 1000))

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(iter(validloader)):
            output = model(inputs)
            valid_loss = criterion(output, labels)
            running_valid_loss += valid_loss.item()

            if i % 1000 == 999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (e + 1, i + 1, running_valid_loss / 1000))

    epoch_values.append(e+1)    
    loss_values.append(running_loss/2000)
    valid_loss_values.append(running_valid_loss/1000)

plt.plot(epoch_values, loss_values)
plt.plot(epoch_values, valid_loss_values)
# plt.legend('')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.xlim((1, epochs))
plt.show()
