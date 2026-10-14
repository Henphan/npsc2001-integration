import cv2 as cv
import os
from pathlib import Path

# DUPLICATE THE MASK INTO 50 DIFFERENT NAMES

PATH_TO_IMAGES = Path("/home/henry/Code/images/")
PATH_TO_MASKS = Path("/home/henry/Code/masks/")

# READ THE MASK AS IMAGE

mask = Path(list(PATH_TO_MASKS.iterdir())[0])
mask_img = cv.imread(mask)
if mask_img is None:
    raise Exception("Image is not valid: {}".format(str(mask)))

# READING THE IMAGES

images = PATH_TO_IMAGES.iterdir()

for image in images:
    identifier = image.name.split(".")[0]
    new_name = "{}_vessel.png".format(identifier)
    final_path = PATH_TO_MASKS / new_name
    cv.imwrite(final_path, mask_img)
