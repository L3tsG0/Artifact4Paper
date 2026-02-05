import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import PIL
import timm
import cv2
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional

from PIL import Image

np.random.seed(42)

from matplotlib import style
from typing import Final
from enum import Enum

class DatasetTypes(Enum):
    ARTS = "ARTS"
    GTSRB = "GTSRB"
    

@dataclass
class ResponseFromClassifier:
    name: str
    label: str
    prob: float
    model: str

    @classmethod
    def from_dict(cls, data: dict) -> 'ResponseFromClassifier':
        return cls(
            name=data['name'],
            label=data['label'],
            prob=data['prob'],
            model=data['model']
        )

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'label': self.label,
            'prob': self.prob,
            'model': self.model
        }

DATA = DatasetTypes.ARTS.value


device = 'cuda'

image_size = 60
IMG_HEIGHT:Final[int] = 60
IMG_WIDTH:Final[int] = 60

NUM_CATEGORIES:Final[int] = 43


class GtsrbDataset(Dataset):
    def __init__(self, df, mode, transform=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.paths = df['path'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img = np.array(PIL.Image.open(self.paths[index]).convert('RGB'))
        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        img = img.transpose(2, 0, 1)
        label = self.labels[index]
        return img, label
    
class SimpleCNN(nn.Module):
    def __init__(self, out_dim):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            
            nn.Flatten(),
            nn.Linear(18432, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits


transforms_valid = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
    albumentations.Normalize()
])

#valid_dataset = GtsrbDataset(df_train, 'valid', transform=transforms_valid)

from torch import nn

class KernelType(Enum):
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    DENSENET121 = "densenet121"
    EFFICIENTNET_B0 = "efficientnet_b0"
    SimpleCNN = "SimpleCNN"
    

def select_model(kernel_type:KernelType):
    


    map_models = {}
    
    if kernel_type == KernelType.SimpleCNN:
        model = SimpleCNN(NUM_CATEGORIES)
        data = torch.load(f'/home/tsuruoka/hdd/blender_file/traffic_sign_classifiers/models_60/SimpleCNN_{DATA}_best_loss.pth', map_location=torch.device('cpu'))
        model.load_state_dict(data['model'])
            
        classes_GTSRB = data['classes']
        model = model.to(device).eval()



        map_models[kernel_type.value] = model, classes_GTSRB

        
        return map_models
        
    
    else:
        
        timm_model = timm.create_model(kernel_type.value, pretrained=False)
        try:
            timm_model.fc = nn.Linear(timm_model.fc.in_features, NUM_CATEGORIES)
        except:
            timm_model.classifier = nn.Linear(timm_model.classifier.in_features, NUM_CATEGORIES)
        timm_model.eval()
        data = torch.load(f'/home/tsuruoka/hdd/blender_file/traffic_sign_classifiers/models_60/{kernel_type.value}_{DATA}_best_loss.pth', map_location=torch.device('cpu'))
        timm_model.load_state_dict(data['model'])
        map_models[kernel_type.value] = timm_model.to(device), data['classes']
        
        return map_models
    
    
import select
import scipy
def read_from_opencv_and_preprocess(img_path:str):
    image = cv2.imread(img_path)
    stop_sign = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    at_image = stop_sign
    res = transforms_valid(image=at_image)
    img = res['image']
    img = torch.tensor(img.transpose(2, 0, 1))#.unsqueeze(0)
    img = torch.stack([img], axis=0)
    return img, at_image

def process_classifier(img:torch.Tensor,at_image:np.ndarray,img_path:str,map_models:dict):
    
    map_res = {}
    all_res = []
    text = '/'.join(img_path.split('/')[-2:])
    name_ = str(text)

    kernel_type_name, (model, classes_GTSRB) = next(iter(map_models.items()))
    pred = model(img).softmax(axis=1).cpu().detach().numpy()
    label = pred.argmax(axis=1)[0]

    GROUND_TRUTH = label

    text += f'\n{DATA} {kernel_type_name}: {classes_GTSRB[label]} ({pred.max():.2f})'
    map_res = dict(name=name_, label=classes_GTSRB[label], prob=pred.max(), model=kernel_type_name)

    all_res.append(map_res)
    return all_res, text,pred
def inference_classifier(img_path:str,kernel_type:KernelType)->ResponseFromClassifier:
    

    preprocessed_img,orig_image = read_from_opencv_and_preprocess(img_path)
    preprocessed_img = preprocessed_img.to(device)

    all_res, text,pred = process_classifier(preprocessed_img,orig_image,img_path,map_models=select_model(kernel_type))
    plt.imshow(orig_image)
    plt.title(text)
    plt.axis('off')

    plt.show()
    
    return all_res[0]
    
def inference_classifier_opencv(image,kernel_type:KernelType)->ResponseFromClassifier:
    

    stop_sign = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    at_image = stop_sign
    res = transforms_valid(image=at_image)
    img = res['image']
    img = torch.tensor(img.transpose(2, 0, 1))#.unsqueeze(0)
    img = torch.stack([img], axis=0)
    img = img.to(device)

    all_res, text,pred = process_classifier(img,img,"",map_models=select_model(kernel_type))

    
    return all_res[0]
    
__all__ = ['KernelType', 'inference_classifier_opencv']


if __name__ == '__main__': 

    a = inference_classifier('/home/tsuruoka/hdd/blender_file/output3.png',KernelType.SimpleCNN)
    print(a)