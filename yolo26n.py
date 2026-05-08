"""
    Importación de paquetes

    cv2: Se utiliza para la lectura y manipulación de imágenes, así como para la visualización de los resultados de detección.
    json: Se emplea para cargar los archivos JSON que contienen las anotaciones de las imágenes en el conjunto de datos.
    os: Se utiliza para la gestión de archivos y directorios, como la creación de carpetas y la copia de archivos.
    pathlib: Se utiliza para recorrer la carpeta de imágenes para encontrar todas, estén en subcarpetas o no.
    shutil: Se emplea para copiar archivos de imágenes a las carpetas correspondientes después de reorganizar el conjunto de datos.
    torch: Permite el uso de la GPU para acelerar el entrenamiento y la inferencia del modelo.
    ultralytics: Proporciona la implementación, entrenamiento y validación del modelo YOLO, que se utiliza para la detección de vehículos.
"""
import cv2
import json
from pathlib import Path
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO


class BDDDatasetReorganizer:
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
        

def getYOLOModel():
    return YOLO("yolo26n").to(device)        


if __name__ == '__main__':

    classes = ['car', 'truck', 'bus', 'train', 'motor', 'bike']

    dr = BDDDatasetReorganizer(
        images_path = "./bdd100k/",
        labels_path = "./bdd100k_labels_release/bdd100k/labels/",
        classes_names = classes,
        dst_path = "./datasets/bdd100k_yolo/"
    )

    if os.path.exists("./datasets/bdd100k_yolo/train") == False:
        dr.organize()

    # Initial cleanup and setup
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    vehicle_det_model = getYOLOModel()

    # Train + Evaluation on the model
    results_vehicle_det = vehicle_det_model.train(data = "./datasets/bdd100k_yolo/train1.yaml",
                              epochs = 50,
                              patience = 25,
                              batch = 64,
                              save_period = 25,
                              cache = True,
                              device = device,
                              workers = 1,     
                              name = "y26_v",
                              exist_ok = True,
                              pretrained = False,
                              optimizer = "auto",
                              verbose = True,
                              multi_scale = 0.25,
                              cos_lr = True,
                              weight_decay = 0.0001,
                              freeze = 0,                            
                              plots = True,
    )

    license_plate_det_model = getYOLOModel()
    results_license_plate_det = license_plate_det_model.train(data = "./datasets/UC3M-LP/train1.yaml",
                              epochs = 10,
                              patience = 25,
                              batch = 64,
                              save_period = 25,
                              cache = True,
                              device = device,
                              workers = 1,     
                              name = "y26_lp",
                              exist_ok = True,
                              pretrained = False,
                              optimizer = "auto",
                              verbose = True,
                              multi_scale = 0.25,
                              cos_lr = True,
                              weight_decay = 0.0001,
                              freeze = 0,                            
                              plots = True,
    )