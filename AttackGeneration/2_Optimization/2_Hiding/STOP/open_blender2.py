import subprocess
import math
import os
import random

blender_path = "<PATH_TO_PROJECT_ROOT>/blender-4.1.1-linux-x64/blender"  # Specify path to Blender executable
file_path = "<PATH_TO_PROJECT_ROOT>/blender_file/hiding/STOP/20240506_hiding_attacker_multiple_color2.blend"
# Build the Blender launch command
script_path = "<PATH_TO_PROJECT_ROOT>/blender_file/hiding/STOP/hiding_attacker2.py"

Patch = {"NittoL":{"red_patch_roughness":0.43,"red_patch_retroreflectivity":15},"HIP3090":{"red_patch_roughness":0.345,"red_patch_retroreflectivity":45.},"Nikkalite":{"red_patch_roughness":0.45,"red_patch_retroreflectivity":105}
,"DG4090":{"red_patch_roughness":0.399,"red_patch_retroreflectivity":125.}}

patchname: list[str] = ["DG4090"]
patch_surface: list[float] = [1/16,1/8,3/16,1/4]

background_image_dir:str = "<PATH_TO_PROJECT_ROOT>/blender_file/night_val"


for patch in patchname:
    for surface in patch_surface:
        print(f"patch: {patch}, surface: {surface}")
        image_list = os.listdir(background_image_dir)
        random.shuffle(image_list)
        image_list = image_list[:100]

        for image in image_list:

            command: list[str] = [
                blender_path,
                file_path,
                "--background",
                "--python", script_path,
                "--",
                "--patch_name", f"{patch}",
                "--max_patch_size", f"{math.sqrt(surface)}",
                "--red_patch_roughness", str(Patch[patch]["red_patch_roughness"]),
                "--red_patch_retroreflectivity", str(Patch[patch]["red_patch_retroreflectivity"]),
                "--background_image_path", str(os.path.join(background_image_dir, image)),
            ]

            # Launch Blender and run the Python script
            subprocess.run(command)