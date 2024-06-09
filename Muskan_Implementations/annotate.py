import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Load pre-trained YOLO model
print("Loading YOLO model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("Model loaded.")

# Path to your dataset folder
dataset_path = "c:/Users/w10/Desktop/drdoimplementation/dataset"  # Replace with your actual dataset path

# Output directory for annotation files
output_dir = "c:/Users/w10/Desktop/drdoimplementation/annotations"  # Replace with your desired output directory

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Loop through each class subfolder and each image in the subfolders
for class_folder in tqdm(os.listdir(dataset_path)):
    class_folder_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_folder_path):  # Ensure it's a directory
        for image_file in os.listdir(class_folder_path):
            if image_file.endswith(".jpg") or image_file.endswith(".JPEG"):  # Assuming images are in jpg or png format
                # Read image
                image_path = os.path.join(class_folder_path, image_file)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Failed to load image {image_path}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (required by YOLO)

                # Run inference on the image
                results = model(image)

                # Print results to debug
                print(f"Detections for {image_file}: {results.xyxy[0]}")
                
                # Check if there are any detections
                detections = results.xyxy[0]
                if len(detections) == 0:
                    print(f"No detections for image {image_file}")
                    continue

                # Create class-specific output directory
                class_output_dir = os.path.join(output_dir, class_folder)
                Path(class_output_dir).mkdir(parents=True, exist_ok=True)

                # Convert detections to YOLO format and save annotations
                annotation_file_path = os.path.join(class_output_dir, os.path.splitext(image_file)[0] + ".txt")
                print(f"Saving annotations for {image_file} to {annotation_file_path}")

                with open(annotation_file_path, "w") as f:
                    for detection in detections:  # Iterate through detected objects
                        class_index = int(detection[5])  # Class index
                        x_center = (detection[0] + detection[2]) / 2 / image.shape[1]  # Normalized x center
                        y_center = (detection[1] + detection[3]) / 2 / image.shape[0]  # Normalized y center
                        width = (detection[2] - detection[0]) / image.shape[1]  # Normalized width
                        height = (detection[3] - detection[1]) / image.shape[0]  # Normalized height
                        f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

print("Annotation process completed.")
