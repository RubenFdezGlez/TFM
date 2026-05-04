# Import necessary libraries
from pathlib import Path

import cv2
import json
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import ultralytics
from ultralytics import YOLO
import yaml

# Reorganizer class explicit for BDD100K on Kaggle
class DatasetReorganizer:
    def __init__(self, images_path, labels_path, classes_names, dst_path):
        self.images_path = images_path
        self.labels_path = labels_path
        self.classes_names = classes_names
        self.dst_path = dst_path


    def createFolders(self):

        for split in ["train", "test", "val"]:
            os.makedirs(os.path.join(self.dst_path, split), exist_ok=True)
            os.makedirs(os.path.join(self.dst_path, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.dst_path, split, "labels"), exist_ok=True)


    def searchImage(self, name, base_path):
        path = Path(base_path)

        for path in path.rglob(name):
            return path
        return None

    def splitDataset(self, data, split):
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

    dr = DatasetReorganizer(
        images_path = "./bdd100k/",
        labels_path = "./bdd100k_labels_release/bdd100k/labels/",
        classes_names = classes,
        dst_path = "./datasets/"
    )

    if os.path.exists("./datasets/train") == False:
        dr.organize()

    # Initial cleanup and setup
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = YOLO("yolo26n").to(device)

    # Train + Evaluation on the model
    results_val = model.train(data = "./datasets/train1.yaml",
                              epochs = 100,
                              device = device,
                              workers = 4, 
                              batch = 12,
                              patience = 25,
                              cos_lr = True,
                              optimizer = "auto",
                              pretrained = False,
                              weight_decay = 0.0001,
                              freeze = 5,
                              verbose = True,
                              plots = True,
                              exist_ok = True,
                              save_period = 25,
                              name = "y26_1",
    )