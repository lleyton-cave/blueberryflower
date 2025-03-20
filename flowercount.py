import os
import sys
from PIL import Image
import pillow_heif
import cv2
import ultralytics
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import re

def heic_to_jpg(input_path, output_path):
    # Register HEIF plugin
    pillow_heif.register_heif_opener()

    # Open HEIC file
    image = Image.open(input_path)

    # Save as JPEG
    image.save(output_path, "JPEG")
    print(f"Converted {input_path} to {output_path}")

    # Remove the HEIC file after conversion
    os.remove(input_path)
    print(f"Removed {input_path} after conversion.")

def convert_all_heic_to_jpg(input_folder):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.heic'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(input_folder, f"{os.path.splitext(filename)[0]}.jpg")  # Save in the same folder
            heic_to_jpg(input_path, output_path)

def rename_photos(input_folder, date_prefix, rename):
    if rename:
        # Get the list of image files (jpg, png, jpeg) in the folder
        images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg','.jpeg','.JPG','.png'))]
        images.sort()  # Sort images alphabetically
    
        # Locate the names.txt file in the folder
        names_file = os.path.join(input_folder, "labels.txt")
        if not os.path.isfile(names_file):
            print("Error: labels.txt file not found.")
            sys.exit(1)

        # Read the new names from the file
        with open(names_file, 'r') as file:
            new_names = file.read().splitlines()
   
        # Check if the number of images and names match
        if len(images) != len(new_names):
            print("Error: The number of images and names do not match.")
            sys.exit(1)
        
        # Rename the photos
        for i, image in enumerate(images):
            old_path = os.path.join(input_folder, image)
            extension = os.path.splitext(image)[1]  # Get file extension
       
            # Create new file name with the date prefix and same extension
            new_name = f"{date_prefix}_{new_names[i]}{extension}"
            new_path = os.path.join(input_folder, new_name)
       
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed '{image}' to '{new_name}'")
   
        print("All photos have been renamed successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python flowercount.py <folder_path> <date> <model_path> <output_bbox> <rename>")
        sys.exit(1)
   
    input_folder = sys.argv[1]
    date_prefix = sys.argv[2]
    model_path = sys.argv[3]
    output_bbox = sys.argv[4].lower() == 'true'  # Convert string to boolean
    rename = sys.argv[5].lower() == 'true'
   
    if not os.path.exists(input_folder):
        print("Error: Folder path does not exist.")
        sys.exit(1)

    if not os.path.isfile(model_path):
        print("Error: Model file does not exist.")
        sys.exit(1)
   
    # Validate date format (YYYYMMDD)
    try:
        datetime.strptime(date_prefix, "%Y%m%d")
    except ValueError:
        print("Error: Date must be in YYYYMMDD format.")
        sys.exit(1)
        
    # Convert HEIC to JPG and remove HEIC files after conversion
    convert_all_heic_to_jpg(input_folder)
    
    # Rename photos
    rename_photos(input_folder, date_prefix, rename)
    
    # Load YOLO model
    model = YOLO(model_path)
    #print(model)
    
    # Create output folder if it doesn't exist
    results_folder = os.path.join(input_folder, "results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Create DataFrame for results
    df = pd.DataFrame(columns=['Date', 'ID', 'flowercount'])

    # Get the list of image files
    image_files = [f for f in os.listdir(os.path.join(input_folder)) if f.endswith(('.jpg', '.jpeg','.JPG','.png'))]
    total_files = len(image_files)
    print(total_files)

    # Loop through images in the output folder
    for idx, filename in enumerate(image_files, start=1):
        # Print progress
        print(f"Processing file {idx} of {total_files}: {filename}")
        
        image_path = os.path.join(input_folder, filename)
        results = model(image_path, verbose=False)

        # Draw bounding boxes if enabled
        if output_bbox:
            img = cv2.imread(image_path)
            rectangle_thickness = 4
            text_thickness = 1
            for result in results:
                for box in result.boxes:
                    cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                  (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                                  (0, 0, 255), rectangle_thickness)
                    cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 128, 0), text_thickness)
            # Save the image with bounding boxes
            output_image_path = os.path.join(results_folder, f"output_{filename}")
            cv2.imwrite(output_image_path, img)

        # Count number of flowers detected
        flower_n = len(result.boxes)
        print(f"Flower Number for {filename} =", flower_n)

        # Add the result to the DataFrame
        row = [{'Date': date_prefix, 'ID': filename, 'flowercount': flower_n}]
        new_row = pd.DataFrame(row)
        df = pd.concat([df, new_row], ignore_index=True)

    # Write the DataFrame to a CSV file in the results folder
    output_csv_path = os.path.join(results_folder, "flower_count_raw.csv")
    df.to_csv(output_csv_path, index=False)
    
    print("Processing complete. Results saved to:", results_folder)
