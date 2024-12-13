import os
import random

def get_original_image_paths(original_path):
    """Get paths for all original images."""
    original_image_paths = []
    for class_name in os.listdir(original_path):
        class_path = os.path.join(original_path, class_name)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
            original_image_paths.extend(images)
    return original_image_paths

def get_corrupted_image_paths(corruption_base_path):
    """Get paths for all corrupted images."""
    corrupted_image_paths = []
    for corruption_type in os.listdir(corruption_base_path):
        corruption_type_path = os.path.join(corruption_base_path, corruption_type)
        if os.path.isdir(corruption_type_path):
            for severity_level in os.listdir(corruption_type_path):
                severity_level_path = os.path.join(corruption_type_path, severity_level)
                if os.path.isdir(severity_level_path):
                    for class_name in os.listdir(severity_level_path):
                        class_path = os.path.join(severity_level_path, class_name)
                        if os.path.isdir(class_path):
                            images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
                            corrupted_image_paths.extend(images)
    return corrupted_image_paths

def pair_images(original_images, corrupted_images):
    """Pair original and corrupted images based on image_id."""
    paired_images = []
    original_image_map = {os.path.basename(img).split('_')[1]: img for img in original_images}
    for corrupted_image in corrupted_images:
        image_id = os.path.basename(corrupted_image).split('_')[1].split('.')[0]  # Extract image_id from corrupted image
        if image_id in original_image_map:
            paired_images.append((original_image_map[image_id], corrupted_image))
    return paired_images

# Example paths
original_path = r'C:\Users\w10\Desktop\DRDO\datasets\Original_Dataset\Original_Dataset'
corruption_base_path = r'C:\Users\w10\Desktop\DRDO\datasets\ImageNet-C-Bar\ImageNet-C-Bar'

# Step 1: Get all original and corrupted image paths
original_images = get_original_image_paths(original_path)
corrupted_images = get_corrupted_image_paths(corruption_base_path)

# Step 2: Pair the images based on image_id
paired_images = pair_images(original_images, corrupted_images)

# Step 3: Print some pairs for verification
for original, corrupted in random.sample(paired_images, 5):  # Display 5 random pairs
    print(f"Original: {original}, Corrupted: {corrupted}")
