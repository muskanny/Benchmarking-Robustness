import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from lpips import LPIPS
import random  # Make sure random module is imported

device = torch.device("cpu")  # Use CPU

# Initialize the LPIPS model
lpips_model = LPIPS(net='alex').to(device)

# Helper function to load and transform ImageNet images (original images)
def load_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        print(f"Successfully loaded image: {image_path}")  # Debugging statement
    except Exception as e:
        print(f"Error loading image: {image_path} - {e}")
        return None
    
    transform = transforms.Compose([ 
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    
    img_tensor = transform(img)
    
    # Check if the tensor shape is correct (3, 224, 224)
    if img_tensor.shape != (3, 224, 224):
        print(f"Warning: Image tensor shape is {img_tensor.shape}. Expected (3, 224, 224).")
    
    return img_tensor.unsqueeze(0)

def calculate_lpips(image1, image2):
    """
    Calculate the LPIPS score between two images.
    """
    image1 = image1.to(device)
    image2 = image2.to(device)
    
    # print(f"Image1 tensor: {image1.shape}, Image2 tensor: {image2.shape}")  # Debugging statement
    
    # Check for NaN values in the image tensors
    if torch.isnan(image1).any() or torch.isnan(image2).any():
        print("NaN found in one of the images!")
        return None
    
    try:
        # Calculate LPIPS score
        lpips_score = lpips_model.forward(image1, image2).item()
        print(f"LPIPS Score: {lpips_score}")  # Debugging statement
    except Exception as e:
        print(f"Error calculating LPIPS score: {e}")
        return None
    
    # Check if the LPIPS score is valid (not NaN or infinite)
    if np.isnan(lpips_score) or np.isinf(lpips_score):
        print("LPIPS score is NaN or infinite!")
        return None
    
    return lpips_score

# Helper function to get corrupted image paths for each severity level, considering only the available classes
def get_corrupted_image_paths(corruption_path, severity_level, corruption_type, valid_classes):
    """
    Get the paths of the corrupted images for a specific corruption type and severity level.
    Only images from valid classes are included.
    """
    corruption_severity_path = os.path.join(corruption_path, str(severity_level))
    image_paths = []
    
    for class_name in os.listdir(corruption_severity_path):
        if class_name not in valid_classes:
            continue  # Skip classes not in the original dataset

        class_path = os.path.join(corruption_severity_path, class_name)
        print(class_path)
        if os.path.isdir(class_path):
            # Retrieve 15 images per class, modify this based on your dataset structure
            images = sorted([os.path.join(class_path, f) for f in os.listdir(class_path)])[:15]
            image_paths.extend(images)
            # print(image_paths)
    return image_paths

def compute_lpips_for_corruption(imagenet_path, corruption_base_path, valid_classes):
    """
    Compute LPIPS scores for all corruptions and severity levels.
    Return the average LPIPS score for each corruption type.
    """
    lpips_scores = []
    
    # Iterate over each corruption type (adjust as per your dataset structure)
    for corruption_type in os.listdir(corruption_base_path):
        corruption_type_path = os.path.join(corruption_base_path, corruption_type)
        if os.path.isdir(corruption_type_path):
            corruption_scores = []
            
            # Iterate over severity levels (1-5)
            for severity_level in range(1,6):  # Severity levels: 1 to 5
                corrupted_image_paths = get_corrupted_image_paths(corruption_type_path, severity_level, corruption_type, valid_classes)
                
                if len(corrupted_image_paths) < 1:
                    print(f"Skipping corruption type {corruption_type} severity {severity_level} due to insufficient images, that is ", len(corrupted_image_paths))
                    continue  # Ensure we have enough images per severity level
                
                # Compute LPIPS for each pair of original and corrupted images
                severity_scores = []
                for corrupted_image_path in corrupted_image_paths:
                    # Assuming original images are stored with a similar structure
                    # print(corrupted_image_path)
                    filedir = os.path.dirname(corrupted_image_path)
                    class_name = os.path.basename(filedir)
                    # print(class_name)
                    
                    original_image_path = os.path.join(imagenet_path, class_name, os.path.basename(corrupted_image_path))
                    
                    # Check if the original image exists
                    if not os.path.exists(original_image_path):
                        print(f"Original image not found: {original_image_path}")  # Debugging statement
                        continue  # Skip to next iteration if the image is missing
                    
                    # Load images
                    original_image = load_image(original_image_path)
                    corrupted_image = load_image(corrupted_image_path)
                    
                    # Skip if images are not loaded correctly
                    if original_image is None or corrupted_image is None:
                        print(f"Skipping pair: {original_image_path}, {corrupted_image_path}")  # Debugging statement
                        continue
                    
                    # Calculate LPIPS score
                    lpips_score = calculate_lpips(original_image, corrupted_image)
                    if lpips_score is not None:
                        severity_scores.append(lpips_score)
                
                # Average LPIPS score for this severity level
                if severity_scores:
                    avg_severity_score = np.mean(severity_scores)
                    corruption_scores.append(avg_severity_score)
            
            # Average LPIPS score for this corruption type
            if corruption_scores:
                avg_corruption_score = np.mean(corruption_scores)
                lpips_scores.append(avg_corruption_score)
    
    return lpips_scores


# Set paths to your original ImageNet dataset and ImageNet-C dataset (corruptions)
imagenet_path = 'C:\\Users\\w10\\Desktop\\DRDO\\datasets\\Original_Dataset\\Original_Dataset'

corruption_base_path = 'C:/Users/w10/Desktop/DRDO/datasets/ImageNet-C-Bar/ImageNet-C-Bar'

# Get the list of valid classes (from your original dataset)
valid_classes = {"n01818515","n01843383","n01883070","n01910747","n01968897", "n02006656","n02037110","n02058221" }
print(valid_classes)
# Compute the LPIPS scores
lpips_results = compute_lpips_for_corruption(imagenet_path, corruption_base_path, valid_classes)

# Print the average LPIPS score for each corruption
print("Average LPIPS scores for each corruption:", lpips_results)

# Final average LPIPS score across all corruptions
if lpips_results:
    final_avg_lpips = np.mean(lpips_results)
    print("Final Average LPIPS Score across all corruptions:", final_avg_lpips)
else:
    print("No valid LPIPS scores were computed.")
