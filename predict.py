import torch
from models import model 
import cv2
from PIL import Image
import os
from torchvision import transforms

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
# print(accuracy_test(testloader=test_dataset))

def predictor(image,model):

    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

    
    print(test_transform(image).unsqueeze(0).shape)
    save_path = 'SatModel.pth'
    # model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    # Generate prediction
    rps_class = model(test_transform(image).unsqueeze(0))
    _, predicted = torch.max(rps_class.data,1)
    
    softmax = torch.nn.Softmax(dim=1)
    ps = softmax(rps_class)
    class_names = {key:val for key, val in enumerate(os.listdir('data/dataset_splits/test'))}
    print(class_names)
    # Positions 
    # position1 = rps_class.values[0]
    # position2 = rps_class.values[-1]
    
    # Annotate 
    # annotate = cv2.rectangle(image,position1,position2,(0,255,0),2)
    # text = cv2.putText(image,f'{rps_class}',position1,cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2)
    # print(rps_class)
    return rps_class.shape, predicted, ps

test_image = Image.open('test_images/map2.png')
rgb = test_image.convert('RGB')
print(rgb.size)
print(predictor(rgb, model))



# predictor(image,model)
# cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
# Input
# What is the output of rps_class
# FUnction receives an image 
# Runs prediction model on image
# Returns image annotated with object
#import the model from RPS_net.pth
# Create dict inside predictor with  key -label names, values-accuracy and the position of pixels in image
# If accuracy is good gonna add class label and add position of label in picture
# Tools we will need
# To annotate -cv2.rectangle(image,position1,position2,color,thickness)
#To get positions - if accuracy score of object>0.6 then gonna add first numpy array values position as position1,
# and last values as position2 

# Input
# FUnction receives an image 
# Runs prediction model on image
# Returns image annotated with object
#import the model from RPS_net.pth
# Create dict inside predictor with  key -label names, values-accuracy and the position of pixels in image
# If accuracy is good gonna add class label and add position of label in picture
