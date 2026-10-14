from pathlib import Path
import subprocess

PATH_TO_IMAGES = Path("/home/henry/Code/images/")
# REMOVE .tif FROM ALL IMAGES IN /images
for file in PATH_TO_IMAGES.iterdir():
    tif_removed = file.stem.split('.')[0]+".png"
    new_file = PATH_TO_IMAGES / tif_removed
    subprocess.run(["cp", file, new_file])

