"""
    Package imports:

    cv2: Reading and image manipulation, as well as drawing bounding boxes on images for visualization.
    json: Loading JSON files containing image annotations in the dataset.
    os: File and directory management, including folder creation and file copying.
    pathlib: Traversing the image folder to find all files, regardless of subfolder structure.
    shutil: Copying image files to corresponding folders after reorganizing the dataset.
"""
import cv2
import json
import os
from pathlib import Path
import shutil


class BDDDatasetReorganizer:
    """Class for reorganizing the BDD100K dataset into YOLO format."""
    def __init__(self, images_path, labels_path, classes_names, dst_path):
        """Initialize the reorganizer with paths and class names.
        
        Args:
            images_path (str): Path to the original images.
            labels_path (str): Path to the original labels in JSON format.
            classes_names (list): List of class names to be included in the dataset.
            dst_path (str): Path to the destination folder where the reorganized dataset will be stored
        """
        self.images_path = images_path
        self.labels_path = labels_path
        self.classes_names = classes_names
        self.dst_path = dst_path


    def createFolders(self):
        """Create the necessary folders for the YOLO dataset."""
        for split in ["train", "test", "val"]:
            os.makedirs(os.path.join(self.dst_path, split), exist_ok=True)
            os.makedirs(os.path.join(self.dst_path, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.dst_path, split, "labels"), exist_ok=True)


    def searchImage(self, name, base_path):
        """Search for an image file with the given name in the dataset base path and return its path.
        
        Args:
            name (str): Name of the image file to search for.
            base_path (str): Base path to search for the image file.
        """
        path = Path(base_path)

        for path in path.rglob(name):
            return path
        return None


    def splitDataset(self, data, split):
        """Extract images and labels from the dataset and save them in YOLO format.

        Args:
            data (dict): Dictionary containing image annotations and labels.
            split (str): Dataset split (train, test, val) to which the images and labels belong.
        """
        self.createFolders()

        for img in data:
            name = img["name"]
            labels = img["labels"]

            src_path = self.searchImage(name, self.images_path)
            if src_path is not None:
                dst_path = os.path.join(self.dst_path, split, "images", name)
                shutil.copy(src_path, dst_path)

                img = cv2.imread(src_path)
                if img is None:
                    print(f"Warning: Could not read image {src_path}")
                    continue
                img_height, img_width = img.shape[:2]

                for label in labels:
                    category = label["category"]
                    if category in self.classes_names:
                        class_id = self.classes_names.index(category)
                        x_center = (label["box2d"]["x1"] + label["box2d"]["x2"]) / 2 / img_width
                        y_center = (label["box2d"]["y1"] + label["box2d"]["y2"]) / 2 / img_height
                        width = (label["box2d"]["x2"] - label["box2d"]["x1"]) / img_width
                        height = (label["box2d"]["y2"] - label["box2d"]["y1"]) / img_height

                        label_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
                        label_path = os.path.join(self.dst_path, split, "labels", name.replace(".jpg", ".txt"))
                        with open(label_path, "a") as label_file:
                            label_file.write(label_line)


    def organize(self):
        """Extracts the data from the JSON files and passes it to the splitDataset method to reorganize the dataset into YOLO format."""
        train_data = {}
        val_data = {}

        try:
            with open(os.path.join(self.labels_path, "bdd100k_labels_images_train.json"), "r") as file:
                train_data = json.load(file)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the file.")
        
        try:
            with open(os.path.join(self.labels_path, "bdd100k_labels_images_val.json"), "r") as file:
                val_data = json.load(file)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the file.")
    
        self.splitDataset(train_data, "train")
        self.splitDataset(val_data, "val")

if __name__ == '__main__':

    classes = ['car', 'truck', 'bus', 'train', 'motor', 'bike']

    dr = BDDDatasetReorganizer(
        images_path = "./bdd100k/",
        labels_path = "./bdd100k_labels_release/bdd100k/labels/",
        classes_names = classes,
        dst_path = "./datasets/bdd100k_yolo/"
    )