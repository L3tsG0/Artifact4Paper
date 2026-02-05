import subprocess
import math
import random
import os
import requests
blender_path = "<PATH_TO_PROJECT_ROOT>/blender-4.1.1-linux-x64/blender"  # Specify path to Blender executable
file_path = "<PATH_TO_PROJECT_ROOT>/SL65/20240615_us_sl65_reflector_multiple_color.blend"
# Build the Blender launch command
script_path = "<PATH_TO_PROJECT_ROOT>/SL65/alternate_attacker2.py"

Patch = {"NittoL":{"white_patch_roughness":0.0294,"white_patch_retroreflectivity":40},"HIP3090":{"white_patch_roughness":0.4667,"white_patch_retroreflectivity":250},"Nikkalite":{"white_patch_roughness":0.4667,"white_patch_retroreflectivity":700}
,"DG4090":{"white_patch_roughness":0.4441,"white_patch_retroreflectivity":570}}
background_image_dir = "<PATH_TO_PROJECT_ROOT>/blender_file/night_val"
patchname = ["Nikkalite","DG4090"]
patch_surface = [1/16,1/8,3/16,1/4]

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
                        "--patch_name", f"{patch}",
                        "--max_patch_size", f"{math.sqrt(surface)}",
                        "--white_patch_roughness", str(Patch[patch]["white_patch_roughness"]),
                        "--white_patch_retroreflectivity", str(Patch[patch]["white_patch_retroreflectivity"]),
                        "--black_patch_roughness", "0.0266",
                        "--black_patch_retroreflectivity", "40",
                        "--background_image_path", str(os.path.join(background_image_dir, image)),
                        "--kernel_type",str(kernel_type)
                            ]

                # Launch Blender and run the Python script
                subprocess.run(command)


