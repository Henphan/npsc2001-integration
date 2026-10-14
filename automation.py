from pathlib import Path
import cv2 as cv
import subprocess
import shutil

PATH_TO_STACKED_MASKS = Path("/home/henry/stacked_oct_scans/")
PATH_TO_MASKS = Path("/home/henry/Italian-Code/masks")
PATH_TO_NEW_MASKS = Path("/home/henry/Italian-Code/new_masks/")

PATH_TO_DATASET = Path("/home/henry/full_dataset/")
PATH_TO_IMAGES = Path("/home/henry/Italian-Code/images/")
PATH_TO_NEW_IMAGES = Path("/home/henry/Italian-Code/new_images/")

PATH_TO_ITALIAN_CODE = Path("/home/henry/Italian-Code/preprocessing.py")

PATH_TO_FINAL_DIR = Path("/home/henry/italian_output/")

# Clearing and re-creating images/
shutil.rmtree(PATH_TO_IMAGES)
PATH_TO_IMAGES.mkdir()
# Clearing and re-creating vessels/
shutil.rmtree(PATH_TO_MASKS)
PATH_TO_MASKS.mkdir()

# Clearing and re-creating new_images/
shutil.rmtree(PATH_TO_NEW_IMAGES)
PATH_TO_NEW_IMAGES.mkdir()
# Clearing and re-creating new_masks/
shutil.rmtree(PATH_TO_NEW_MASKS)
PATH_TO_NEW_MASKS.mkdir()

# Getting a dictionary of identifier:path
# TODO Possible reuse of code
data_dict = dict()
for file in PATH_TO_DATASET.iterdir():
    identifier = file.stem.split('_Z')[0]
    data_dict.setdefault(identifier, []).append(file)

for identifier in data_dict:
    # Setting up images/
    for file in PATH_TO_DATASET.glob(f"{identifier}*"):
        shutil.copy(file, PATH_TO_IMAGES)
    # removing .tif
    for file in PATH_TO_IMAGES.iterdir():
        tif_removed = file.stem.split('.')[0]+".png"
        new_file = PATH_TO_IMAGES / tif_removed
        shutil.move(file, new_file)
    # Setting up masks/
    for file in PATH_TO_STACKED_MASKS.glob(f"{identifier}*"):
        shutil.copy(file, PATH_TO_MASKS)
        img = cv.imread(file)
        if img is None:
            raise Exception("Image is not valid: {}".format(str(mask)))
        # duplicating with different names
        for image in PATH_TO_IMAGES.iterdir():
            identifier_slice = image.name.split(".")[0]
            new_name = "{}_vessel.png".format(identifier_slice)
            final_path = PATH_TO_MASKS / new_name
            cv.imwrite(final_path, img)

    # Running Italian Code
    subprocess.run(
        ["uv", "run", PATH_TO_ITALIAN_CODE]
    )
    # Exporting the outputs

    # making the directory for the identifier
    identifier_dir = PATH_TO_FINAL_DIR / identifier
    if identifier_dir.exists():
        shutil.rmtree(identifier_dir)
    identifier_dir.mkdir()
    # create the sub-directories for new_images/ and new_masks/ 
    new_images_dir = (identifier_dir / "new_images")
    new_masks_dir = (identifier_dir / "new_masks")
    if new_images_dir.exists():
        shutil.rmtree(new_images_dir)
    if new_masks_dir.exists():
        shutil.rmtree(new_masks_dir)
    # new_images_dir.mkdir()
    # new_masks_dir.mkdir()
    # copying the outputs to the directories
    shutil.copytree(PATH_TO_NEW_IMAGES, new_images_dir, dirs_exist_ok = True)
    shutil.copytree(PATH_TO_NEW_MASKS, new_masks_dir, dirs_exist_ok = True)

    # Clearing and re-creating new_images/
    shutil.rmtree(PATH_TO_NEW_IMAGES)
    PATH_TO_NEW_IMAGES.mkdir()
    # Clearing and re-creating new_masks/
    shutil.rmtree(PATH_TO_NEW_MASKS)
    PATH_TO_NEW_MASKS.mkdir()

    # Clearing and re-creating images/
    shutil.rmtree(PATH_TO_IMAGES)
    PATH_TO_IMAGES.mkdir()
    # Clearing and re-creating vessels/
    shutil.rmtree(PATH_TO_MASKS)
    PATH_TO_MASKS.mkdir()

# for identifier in PATH_TO_MASKIS.iterdir():
    # Setting up images/



# for identifier in PATH_TO_MASKS.iterdir():
#     # Create direct path to the stacked_oct_scans directory
#     path = PATH_TO_MASKS / identifier
#     if not (path.exists() and path.is_file()):
#         print("Path is invalid: {}".format(path))
#         continue
#     # Copy the directory to ~/Italian-Code/masks, after clearing it
#     shutil.rmtree(PATH_TO_INPUT_MASKS)
#     PATH_TO_INPUT_MASKS.mkdir()
#     shutil.copy(path, PATH_TO_INPUT_MASKS)
#
#
