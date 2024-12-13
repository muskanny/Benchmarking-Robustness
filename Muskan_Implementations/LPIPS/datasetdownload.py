import os

import mindspore_hub as mindspore
import mindspore.dataset as ds
os.environ["DATASET_DIR"]="C:\\Users\\w10\\Desktop\\DRDO\\datasets\\originalimagenet"
# Path to the ImageNet dataset
dataset_dir = os.getenv("DATASET_DIR", "C:\\Users\\w10\\Desktop\\DRDO\\datasets\\originalimagenet")

# Load original images (validation set for LPIPS calculations)
imagenet_dataset = ds.ImageFolderDataset(
    dataset_dir + "/train",
    shuffle=False
)

# Check dataset structure
for data in imagenet_dataset.create_dict_iterator(output_numpy=True):
    print(data["image"].shape, data["label"])
    break
