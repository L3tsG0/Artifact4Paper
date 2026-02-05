import glob
import os
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import PIL
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

import bpy
import time
import math
import sys


np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--patch_name', type=str, help='Dataset to use')
parser.add_argument('--max_patch_size',type=float,default=0.5,help='max patch size')
parser.add_argument('--red_patch_roughness',type=float,help='red patch  roughness')
parser.add_argument('--red_patch_retroreflectivity',type=float,help='red patch retroreflectivity')
parser.add_argument('--background_image_path',type=str,help='background image')
parser.add_argument('--output_csv_dir',type=str,help='output directory')

args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

# change red patch roughness

bpy.data.materials['Material.016'].node_tree.nodes["Principled BSDF.003"].inputs["Roughness"].default_value = args.red_patch_roughness

bpy.data.materials['Material.016'].node_tree.nodes['Value.001'].outputs[0].default_value = args.red_patch_retroreflectivity

# change background image

image_path = args.background_image_path
image_ = bpy.data.images.load(image_path)
bpy.data.scenes['Scene'].node_tree.nodes['Image'].image=image_


from matplotlib import style


DATA = 'ARTS'
# DATA = "GTSRB"
device = 'cuda'


PATH_TO_IMAGE = "<PATH_TO_HOME_GO>/Code/yolov5/Yolov5_AE/Attacker/stopsign_img/Stopsign_cropped.png"
PATH_TO_IMAGE = "<PATH_TO_HOME_GO>/Code/yolov5/Yolov5_AE/Attacker/stopsign_img/SpeedLimit30.png"
image_size = 60
IMG_HEIGHT = 60
IMG_WIDTH = 60

NUM_CATEGORIES = 43

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


transforms_valid = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
    albumentations.Normalize()
])

#valid_dataset = GtsrbDataset(df_train, 'valid', transform=transforms_valid)

map_models = {}

from torch import nn

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


# load classifier
model = SimpleCNN(NUM_CATEGORIES)
kernel_type = 'SimpleCNN'

data = torch.load(f'<PATH_TO_PROJECT_ROOT>/blender_file/traffic_sign_classifiers/models_60/SimpleCNN_{DATA}_best_loss.pth', map_location=torch.device('cpu'))
model.load_state_dict(data['model'])
    
classes_GTSRB = data['classes']
model = model.to(device).eval()
map_models['SimpleCNN'] = model, classes_GTSRB
# STOPSIGN = 14 # GTSRB

STOPSIGN = 12 # ARTS
from datetime import datetime

# inference
def inference(path,export_log=False):
        # Coordinates of the selected region
    x1, y1 = 378, 219
    x2, y2 = 480, 318

    

    image = cv2.imread(path)
    stop_sign = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Crop the image to the selected region
    at_image = stop_sign[y1:y2, x1:x2]
    
    cv2.imwrite(at_image,"<PATH_TO_HOME_GO>/Desktop/crp2.jpg")
    
    # EoT image processing goes here
    
    ##
    
    res = transforms_valid(image=at_image)
    img = res['image']
    img = torch.tensor(img.transpose(2, 0, 1))#.unsqueeze(0)
    img = torch.stack([img], axis=0).to(device)
    map_res = {}

    # text = '/'.join(PATH_TO_IMAGE.split('/')[-2:])
    text = '/'.join(path.split('/')[-2:])
    name_ = str(text)

    all_res = []
    for k, (model, classes_GTSRB) in map_models.items():
        pred = model(img).softmax(axis=1).cpu().detach().numpy()
        label = pred.argmax(axis=1)[0]

        text += f'\n{DATA} {k}: {classes_GTSRB[label]} ({pred.max():.2f})' # model_name,class,(prob)
        prob_of_stopsign = pred[0][STOPSIGN]
        prob = pred.max()
        map_res = dict(name=name_, label=classes_GTSRB[label], prob=pred.max(), model=k)

        all_res.append(map_res)

    print(text)

    with open("<PATH_TO_HOME_GO>/Desktop/test.log", "a")as f:
        f.write(text)
        f.write("\n")
        f.write(f"prob: {prob:.2f}")
        f.write("\n")
        f.write(f"prob of ground truth: {prob_of_stopsign:.2f}")

    return prob,prob_of_stopsign



def save_picture(savepath = "<PATH_TO_HOME_GO>/Code/yolov5/Yolov5_AE/Sato/blender/render/optuna",best=False,RGBA=False)->str:
    
    time_ = datetime.now().strftime("%Y%m%d%H%M%S-%f")
    
    if best:
        time_ = 'best'
    if RGBA:
        time_ = 'best_rgba'
        bpy.context.scene.render.image_settings.color_mode='RGBA'
    bpy.context.scene.render.filepath = f"{savepath}/{time_}.png"
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.render.image_settings.color_mode='RGB'
    
    return bpy.context.scene.render.filepath

# inference
def inference_ae(path,export_log=False,N=10,best = False):
        # Coordinates of the selected region
    x1, y1 = 364,221
    x2, y2 = 462, 319

    

    image = cv2.imread(path)
    stop_sign = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Crop the image to the selected region
    at_image =at_image2= stop_sign[y1:y2, x1:x2]
    at_image_orig = at_image
    
    if best:
        
        at_image2 = cv2.cvtColor(at_image2,cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{path.replace('.png','')}-cropped.png",at_image2)
    


    prob_rireki = []
    stop_prob_rireki = []

    for i in range(N):
        
        blightness_factor = 1
        blightness_factor = np.random.uniform(0.8,2.0)
        
        if i == 0:
            blightness_factor = 1
        print(blightness_factor)

        at_image = np.clip(at_image_orig * blightness_factor, 0, 255).astype(np.uint8)
    
        ##
        
        res = transforms_valid(image=at_image)
        img = res['image']
        img = torch.tensor(img.transpose(2, 0, 1))#.unsqueeze(0)
        img = torch.stack([img], axis=0).to(device)
        map_res = {}

        # text = '/'.join(PATH_TO_IMAGE.split('/')[-2:])
        text = '/'.join(path.split('/')[-2:])
        name_ = str(text)

        all_res = []
        for k, (model, classes_GTSRB) in map_models.items():
            pred = model(img).softmax(axis=1).cpu().detach().numpy()
            label = pred.argmax(axis=1)[0]

            text += f'\n{DATA} {k}: {classes_GTSRB[label]} ({pred.max():.2f})' # model_name,class,(prob)
            prob_of_stopsign = pred[0][STOPSIGN]
            prob = pred.max()
            map_res = dict(name=name_, label=classes_GTSRB[label], prob=pred.max(), model=k)

            all_res.append(map_res)

        print(text)
        
        if np.max(pred[0])-prob_of_stopsign>0:
            at_img2 = cv2.cvtColor(at_image,cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{path}-{i}_{classes_GTSRB[label]} ({pred.max():.2f}.png",at_img2)


            
        with open(f"{dir}/test.log", "a")as f:
            
            if i ==0:
                benign_prob = pred[0]
                f.write('\n')
            f.write(text)
            f.write("\n")
            f.write(f"prob: {prob:.2f}")
            f.write("\n")
            f.write(f"prob of ground truth: {prob_of_stopsign:.2f}\n")
            
            if i == N-1:
                f.write("\n")

        prob_rireki.append(pred[0])
        stop_prob_rireki.append(prob_of_stopsign)
    
    prob = np.mean(prob_rireki,axis = 0)
    prob_of_stopsign = np.mean(stop_prob_rireki)

    return prob,prob_of_stopsign






REFLECTOR = bpy.data.objects["Plane.008"] 
REFLECTOR_W = 0.3
REFLECTOR_H = 0.3







# Change background image of the composite node

# bpy.data.images['Image'].filepath = PATH_TO_IMAGE

# First patch
patch1 = bpy.data.objects['Plane.008']

import optuna
STOPSIGN_SIZE = 1.84
default_y = 0
default_z = 2.64

dir = f"<PATH_TO_PROJECT_ROOT>/blender_file/render/optuna/STOPSIGN/{args.patch_name}/{args.max_patch_size}/STOPSIGN_{args.patch_name}_maxpatchsize_{args.max_patch_size}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"


os.makedirs(dir)

import logging
optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(f'{dir}/test.log'))

def objective(trial):

    REFLECTOR_W1 = trial.suggest_uniform('REFLECTOR_W1', 0.1, args.max_patch_size)
    REFLECTOR_H1 = trial.suggest_uniform('REFLECTOR_H1', 0.1, args.max_patch_size)



    y1 = trial.suggest_uniform('y1', -1.2,1.2)
    z1 = trial.suggest_uniform('z1', 1.4, 3.7)


    # Set patch 1 position and size

    patch1.location = (0.18, y1, z1)

    patch1.scale[0] = REFLECTOR_W1
    patch1.scale[1] = REFLECTOR_H1a

# set_reflector_size(reflector=REFLECTOR,w=REFLECTOR_W,h=REFLECTOR_H) # TODO: change reflector size here.
    sun = bpy.data.objects["Lamp"]  # Sun light (object name remains Lamp)
    _sun_type = sun.data.type
    _sun_energy = sun.data.energy
    _sun_angle = sun.data.angle

    try:
        file_path = save_picture(dir,best = False)

        argmax_prob,stopsign_prob = inference_ae(file_path,export_log=True,N=5)

        # Force Sun type just in case
        sun.data.type = "SUN"
        # Intensity
        sun.data.energy = 0.2
        # Shadow softness (angle is in radians)
        sun.data.angle = math.radians(180)

        file_path = save_picture(dir,best = False)
        _, day_prob = inference_ae(file_path,export_log=True,N=1)

        #    return stopsign_prob
        #    return np.max(argmax_prob)-stopsign_prob

        return stopsign_prob - day_prob
    finally:
        sun.data.type = _sun_type
        sun.data.energy = _sun_energy
        sun.data.angle = _sun_angle

def test(params):
    
    REFLECTOR_W1 = params['REFLECTOR_W1']
    REFLECTOR_H1 = params['REFLECTOR_H1']

   

    y1 = params['y1']
    z1 = params['z1']


    # Set patch 1 position and size

    patch1.location = (0.18, y1, z1)

    patch1.scale[0] = REFLECTOR_W1
    patch1.scale[1] = REFLECTOR_H1

# set_reflector_size(reflector=REFLECTOR,w=REFLECTOR_W,h=REFLECTOR_H) # TODO: change reflector size here.
    file_path = save_picture(dir,best = True)

    prob,stopsign_prob = inference_ae(file_path,export_log=True,N=1,best = True)
    
#    if np.argmax(prob) == 14:
#        return -0.5

    label = prob.argmax()

    print(f'{classes_GTSRB[label]} ({prob.max():.2f})')
    
    print(f"ground_truth ({classes_GTSRB[STOPSIGN]}): {prob[STOPSIGN]}")
    
    with open(f"{dir}/log.txt", "a")as f:
            f.write(f"Optimized params:{params}\n")
            f.write(f'best_attack_result: {classes_GTSRB[label]} ({prob.max():.2f})\n')
            f.write(f"ground_truth ({classes_GTSRB[STOPSIGN]}): {prob[STOPSIGN]}\n")
            f.write(f"each_class_conf is : {prob}")

    import json

    data = {
        "patch_name": args.patch_name,
        "background_img": args.background_image_path,
        "max_patch_size": args.max_patch_size,
        "Optimized params": str(params),
        "Best_attack_result num": f"{label}",
        "Best_attack_result": f"{classes_GTSRB[label]} ({prob.max():.2f})",
        "Ground_truth_num": f"{STOPSIGN}",
        "Ground_truth": f"{classes_GTSRB[STOPSIGN]} ({prob[STOPSIGN]:.2f})",
        "Each class confidence": [float(f"{conf:.2f}") for conf in prob]
    }

    with open(f"{dir}/log.json", "a") as f:
        json.dump(data, f)
        f.write("\n")

    # Write the same content to output_csv_dir

    os.makedirs(args.output_csv_dir,exist_ok=True)


    with open(f"{args.output_csv_dir}/log.json", "a") as f:
        json.dump(data, f)
        f.write("\n")
            
    save_picture(dir,RGBA=True)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print(study.best_params)
test(study.best_params)