import bpy
import subprocess
import json
import os
import re
import sys
import argparse
import time

import math
parser = argparse.ArgumentParser()
parser.add_argument('--patch_name', type=str, help='Dataset to use')
parser.add_argument('--max_patch_size',type=float,default=0.5,help='max patch size')
parser.add_argument('--red_patch_roughness',type=float,help='red patch  roughness')
parser.add_argument('--red_patch_retroreflectivity',type=float,help='red patch retroreflectivity')
parser.add_argument('--background_image_path',type=str,help='background image')
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

# change red patch roughness

bpy.data.materials['Material.016'].node_tree.nodes["Principled BSDF.003"].inputs["Roughness"].default_value = args.red_patch_roughness

bpy.data.materials['Material.016'].node_tree.nodes['Value.001'].outputs[0].default_value = args.red_patch_retroreflectivity

# change background image 

image_path = args.background_image_path
image_ = bpy.data.images.load(image_path)
bpy.data.scenes['Scene'].node_tree.nodes['Image'].image=image_

dir = f"<PATH_TO_PROJECT_ROOT>/blender_file/render/optuna/hiding/STOPSIGN/2/{args.patch_name}/{args.max_patch_size}/STOPSIGN_{args.patch_name}_maxpatchsize_{args.max_patch_size}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"


os.makedirs(dir)


def run_yolov5_detection(render_output_path,filter_name=None):

    import requests

    url = "http://localhost:3141/detect"
    data = {"file_path": f"{render_output_path}"}
    response = requests.post(url, json=data)
    result = response.json()

    detections = result["detections"]

    # Get confidence for R1-1
    if filter_name is not None:
        detection = [d for d in detections if d["name"] == filter_name]
        
        if len(detection) == 0:
            print("No object detected")
            return 0
        
        detection = max(detection, key=lambda x: x["confidence"])
        return detection['confidence']


    if len(detections) == 0:
        print("No object detected")
        return 0

    # Get detections whose name is R1-1

    detection = [d for d in detections if d["name"] == "R1-1"]


    detection = max(detections, key=lambda x: x["confidence"])



    print(detection['confidence'])

    return detection['confidence']

from datetime import datetime
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
    
    
    # Parse stdout as JSON
import numpy as np
import cv2
import scipy.ndimage
# Set image path for rendering output
def render_and_get_confidence(N=10,best=False,RGBA=False):

    sun = bpy.data.objects["Lamp"]  # Sun light (object name remains Lamp)
    _sun_type = sun.data.type
    _sun_energy = sun.data.energy
    _sun_angle = sun.data.angle

    try:
        def _score(render_output_path, N):
            score_sum = 0

            for i in range(N):
                at_image_orig = cv2.imread(render_output_path)
                name, extension = render_output_path.rsplit('.', 1)

                if(i<N//2):
                    blightness_factor = 1
                    blightness_factor = np.random.uniform(0.8,2.0)
                    
                    if i == 0:
                        blightness_factor = 1
                    print(blightness_factor)

                    at_image = np.clip(at_image_orig * blightness_factor, 0, 255).astype(np.uint8)

                elif i<N-1:
                    
                    zoom_factor = np.random.uniform(0.75, 1.25)

                    if i == N//2:
                        zoom_factor = 0.75
                    at_image = scipy.ndimage.zoom(at_image_orig, zoom=(1, zoom_factor, 1), order=1)
                    
                else:
                    at_image = scipy.ndimage.rotate(at_image_orig, angle=15, reshape=False)
                

                new_filename = name + f"_{i}." + extension
                cv2.imwrite(f"{new_filename}", at_image)

                # Run object detection

                score = run_yolov5_detection(f"{new_filename}",filter_name="R1-1")

                # Log scores to log.txt

                with open(f"{dir}/log.txt", "a") as f:
                    f.write(f"{new_filename}: {score}\n")

                print(f"score: {score}")
                score_sum += score

            return score_sum/N

        render_output_path = save_picture(dir,best,RGBA)
        night_score = _score(render_output_path, N)

        # Force Sun type just in case
        sun.data.type = "SUN"
        # Intensity
        sun.data.energy = 0.2
        # Shadow softness (angle is in radians)
        sun.data.angle = math.radians(180)

        render_output_path = save_picture(dir,best,RGBA)
        day_score = _score(render_output_path, 1)

        return night_score - day_score
    finally:
        sun.data.type = _sun_type
        sun.data.energy = _sun_energy
        sun.data.angle = _sun_angle
patch1 = bpy.data.objects['Plane.008']



# Optimize patch position and size with Optuna


def objective(trial):
    REFLECTOR_W1 = trial.suggest_uniform('REFLECTOR_W1', 0.1, args.max_patch_size)
    REFLECTOR_H1 = trial.suggest_uniform('REFLECTOR_H1', 0.1, args.max_patch_size)
    

    y1 = trial.suggest_uniform('y1', -1.2,1.2)
    z1 = trial.suggest_uniform('z1', 1.4, 3.7)


    


    # Set patch 1 position and size

    patch1.location = (0.18, y1, z1)

    patch1.scale[0] = REFLECTOR_W1
    patch1.scale[1] = REFLECTOR_H1
    


    # Render and run object detection
    
    confidence = render_and_get_confidence()
#    print(confidenmce)

    return confidence

def test_params(params):
    REFLECTOR_W1 = params['REFLECTOR_W1']
    REFLECTOR_H1 = params['REFLECTOR_H1']





    y1 = params['y1']
    z1 = params['z1']




    # Set patch 1 position and size

    patch1.location = (0.18, y1, z1)

    patch1.scale[0] = REFLECTOR_W1
    patch1.scale[1] = REFLECTOR_H1


    


    # Render and run object detection
    
    confidence = render_and_get_confidence(N=1)

    with open(f"{dir}/log.txt", "a")as f:
        f.write(f"Optimized params:{params}\n")
        f.write(f'best_attack_result: {confidence}\n')


    data = {
        "patch_name": args.patch_name,
        "background_img": args.background_image_path,
        "max_patch_size": args.max_patch_size,
        "Optimized params": str(study.best_params),
        "Optimized confidence": confidence,
    }

    with open(f"{dir}/log.json", "a") as f:
        json.dump(data, f)
        f.write("\n")

    return confidence

from PIL import Image


# Run Optuna optimization
import optuna
study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials=100)

print(study.best_params)
print(study.best_value)


best_attack_result = test_params(study.best_params)


# Load existing JSON file
with open(f"{dir}/log.json", "r") as f:
    data = json.load(f)
  
best_result_rgba = save_picture(dir,RGBA=True,best=True)
data["best_result_rgba_path"] = best_result_rgba
with open(f"{dir}/log.json", "a") as f:
    json.dump(data, f)
    f.write("\n")
print(f"best_result:{best_attack_result}")













