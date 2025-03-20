Can be used directly from cmd with,

e.g. yolo predict model=/flowermodel.pt source=/images/IMG_4193.jpg

Or can be run on a folder of images with flowercount.py script by,

python flowercount.py <folder_path> <date> <model_path> <output_bbox> <rename>

e.g. python flowercount.py /images YYYYMMDD /flowermodel.pt true false

<folder_path> path to image folder

<date> Date as YYYYMMDD e.g. 20250320

<model_path> path to model

<output_bbox> defines whether to output images in the results with bounding boxes

<rename> defines whether to rename images this can be helpful for images in a set sequence where a walk path can be defined and relies upon a .txt file called labels.txt with each row containing a name for the corresponding image

check flowercount.py for requirements

model was trained from a set of images taken on an iphone 16 and may not generalise well to other sensors and environments 
