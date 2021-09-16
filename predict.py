import torch 
import torch.nn as nn
from models import model 

model.eval()
def accuracy_test(testloader):
    predictions = []
    correct,total = 0,0
    for i,data in enumerate(testloader,0):
        inputs,labels = data
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data,1)
        predictions.append(outputs)
        total +=labels.size(0)

        correct += (predicted == labels).sum().item()

    print('The testing set accuracy of network is %d %%'%(100*correct/total))

def predictor(image,model):
    save_path = 'SatModel.pth'
    # model = Net()
    model = model()
    model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    model.eval()

    # Generate prediction
    rps_class = model(image)

# Input
# FUnction receives an image 
# Runs prediction model on image
# Returns image annotated with object
#import the model from RPS_net.pth
# Create dict inside predictor with  key -label names, values-accuracy and the position of pixels in image
# If accuracy is good gonna add class label and add position of label in picture
