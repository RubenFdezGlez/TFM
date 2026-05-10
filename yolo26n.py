"""
    Package imports:

    torch: Enabling GPU usage to accelerate model training and inference.
    ultralytics: Provides the implementation, training and validation of the YOLO model, which is used for vehicle detection.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
        

def getYOLOModel(device):
    """Load the YOLO model and return it ready for training or inference.
    
    Args:
        device (torch.device): The device (CPU or GPU) to which the model will be sent.

    Returns:
        YOLO: The loaded YOLO model ready for training or inference.
    """
    return YOLO("yolo26n").to(device)        


if __name__ == '__main__':
    
    """Train the YOLO model for vehicle detection and license plate detection."""
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vehicle_det_model = getYOLOModel(device)
    results_vehicle_det = vehicle_det_model.train(data = "./datasets/bdd100k_yolo/train1.yaml",
                              epochs = 50,
                              patience = 10,
                              batch = 64,
                              save_period = 20,
                              cache = True,
                              device = device,
                              workers = 1,     
                              name = "y26_v",
                              exist_ok = True,
                              pretrained = True,
                              optimizer = "auto",
                              verbose = True,
                              multi_scale = 0.0,
                              cos_lr = True,
                              weight_decay = 0.0001,
                              freeze = 0,                            
                              plots = True,
    )

    license_plate_det_model = getYOLOModel(device)
    results_license_plate_det = license_plate_det_model.train(data = "./datasets/UC3M-LP/train1.yaml",
                              epochs = 25,
                              patience = 5,
                              batch = 64,
                              save_period = 10,
                              cache = True,
                              device = device,
                              workers = 1,     
                              name = "y26_lp",
                              exist_ok = True,
                              pretrained = True,
                              optimizer = "auto",
                              verbose = True,
                              multi_scale = 0.0,
                              cos_lr = True,
                              weight_decay = 0.0001,
                              freeze = 0,                            
                              plots = True,
    )