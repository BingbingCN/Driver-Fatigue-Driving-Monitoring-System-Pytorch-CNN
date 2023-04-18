import torch
from PIL import Image
from torchvision import datasets, transforms
import torch.nn as nn
import cv2
import numpy as np

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, 1.3,5 )
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(3 * 3 * 64, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out





train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),

        transforms.ToTensor(),
    ]
)



def detect_face_predict(dst):
    flag=False
    model = CNN()
    model.load_state_dict(torch.load('./result/CNN.pt', map_location='cpu'))
    model.to('cpu')

    cascade = cv2.CascadeClassifier("./xml/haarcascade_frontalface_alt2.xml")
    rects = detect(dst, cascade)
    # print(rects)

    if len(rects)==0:
        print("No face..")
    else:
        for x1, y1, x2, y2 in rects:
            # 调整人脸截取的大小。横向为x,纵向为y
            roi = dst[y1:y2, x1:x2]
            img_roi = roi
            re_roi = cv2.resize(img_roi, (224, 224))



            with torch.no_grad():
                re_roi=Image.fromarray(re_roi)
                input=train_transforms(re_roi).reshape(1,3,224,224)

                if model(input).argmax(dim=-1).cpu().numpy()[0]==1:


                    print('Fatigue..')
                    flag = True
                else:
                    print('Active..')
                    flag = False
    return flag

