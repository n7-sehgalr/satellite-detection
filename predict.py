import torch
from torchvision.transforms.transforms import RandomCrop
from models import model 
import cv2
from PIL import Image
import os
from torchvision import transforms
import numpy as np

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

# def annotation(img1, img, pred_no):
#     dic = {"1":""}
#     img_gray = cv2.CvtColor(img1, cv2.COLOR_RGB2GRAY)
#     template = cv2.CvtColor(img1, cv2.COLOR_RGB2GRAY)

#     w, h = template.shape[::-1]

#     res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

#     thres = 0.8
#     loc = np.where(res >= thres)
#     for i in zip(*loc[::-1]):
#         cv2.rectangle(img1, i, (i[0] + w, i[1] + h), (0,255,255), 3)
#     cv2.putText(img1, dic(pred_no), (w, h), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,0), THICKNESS=2)
#     return img1


def predictor(image,model):

    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

    
    print(test_transform(image).unsqueeze(0).shape)
    save_path = 'trained_model.pth'
    # model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    # Generate prediction
    rps_class = model(test_transform(image).unsqueeze(0))
    _, predicted = torch.max(rps_class.data,1)
    
    softmax = torch.nn.Softmax(dim=1)
    ps = softmax(rps_class)
    class_names = {key:val for key, val in enumerate(os.listdir('data/2750'))}
    print(class_names)
    # Positions 
    # position1 = rps_class.values[0]
    # position2 = rps_class.values[-1]
    
    # Annotate 
    # annotate = cv2.rectangle(image,position1,position2,(0,255,0),2)
    # text = cv2.putText(image,f'{rps_class}',position1,cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2)
    # print(rps_class)
    return rps_class, class_names[int(predicted)], ps

test_image = Image.open('test_images/map2.png')
rgb = test_image.convert('RGB')
print(rgb.size)
print(predictor(rgb, model))
