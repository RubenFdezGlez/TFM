"""
    Package imports:

    torch: Enabling GPU usage to accelerate model training and inference.
    ultralytics: Provides the implementation, training and validation of the YOLO model, which is used for vehicle detection.
"""
import torch
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
    
    """Train the YOLO model for license plate detection."""
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    license_plate_det_model = getYOLOModel(device)
    results_license_plate_det = license_plate_det_model.train(data = "./datasets/UC3M-LP/yolo.yaml",
                              epochs = 25,
                              patience = 5,
                              batch = 12,
                              save_period = 10,
                              cache = True,
                              device = device,
                              workers = 4,     
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