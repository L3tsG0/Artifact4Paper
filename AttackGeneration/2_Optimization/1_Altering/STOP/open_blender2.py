import subprocess
import os
import random
import requests
random.seed(0)

blender_path = "<PATH_TO_PROJECT_ROOT>/blender-4.1.1-linux-x64/blender"  # Specify path to Blender executable
file_path = "/path/to/your/blender_file.blend"  # Path to the .blend file to open (optional)
file_path = "<PATH_TO_PROJECT_ROOT>/blender_file/20240506_alternating_attacker_multiple_color3.blend"
# Build the Blender launch command
script_path = "<PATH_TO_PROJECT_ROOT>/blender_file/alternate_attacker_another_model.py"

background_image_dir = "<PATH_TO_PROJECT_ROOT>/blender_file/night_val"

Patch = {"NittoL":{"red_patch_roughness":0.43,"red_patch_retroreflectivity":15},"HIP3090":{"red_patch_roughness":0.345,"red_patch_retroreflectivity":45.},"Nikkalite":{"red_patch_roughness":0.45,"red_patch_retroreflectivity":105}
,"DG4090":{"red_patch_roughness":0.399,"red_patch_retroreflectivity":125.}}



# Randomly select 100 images from background_image_path and attack each

patchname = ["NittoL","HIP3090","Nikkalite","DG4090"]
patch_surface = [1/16,1/8,3/16,1/4]
import math
for kernel_type in ['resnet50', 'densenet121', 'efficientnet_b0']:
    for patch in patchname:
        for surface in patch_surface:
            
            image_list = os.listdir(background_image_dir)
            random.shuffle(image_list)
            image_list = image_list[:20]

            for image in image_list:




                command = [
                    blender_path,
                    file_path,
                    "--background",
                    "--python", script_path,
                    "--",
                    "--patch_name", patch,
                    "--max_patch_size", f"{math.sqrt(surface)}",
                    "--red_patch_roughness", str(Patch[patch]["red_patch_roughness"]),
                    "--red_patch_retroreflectivity", str(Patch[patch]["red_patch_retroreflectivity"]),
                    "--output_csv_dir", f"attack_effectiveness_evaluation2/{patch}",
                    "--background_image_path", str(os.path.join(background_image_dir, image)),
                    "--kernel_type",str(kernel_type)
                ]

                # Launch Blender and run the Python script
                subprocess.run(command)


# Also randomly select 100 images for NittoL and attack each





