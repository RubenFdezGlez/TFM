"""
    Package imports:

    albumentations: Provides multiple Data Augmentation functions to apply to the dataset.
    torch: Enabling GPU usage to accelerate model training and inference.
"""
import albumentations as A
import numpy as np
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor, EarlyStoppingCallback
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from transformers import TrainingArguments, Trainer


"""

"""
checkpoint = "PekingU/rtdetr_r18vd_coco_o365"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id2label = {0: "car", 1: "truck", 2: "bus", 3: "train", 4: "motor", 5: "bike"}
label2id = {"car": 0, "truck": 1, "bus": 2, "train": 3, "motor": 4, "bike": 5}

model = AutoModelForObjectDetection.from_pretrained(checkpoint,
                                                    id2label=id2label,
                                                    label2id=label2id,
                                                    ignore_mismatched_sizes=True).to(device)
processor = AutoImageProcessor.from_pretrained(checkpoint, backend="torchvision")
# Configure processor for the target size
processor.size = {"height": 640, "width": 640}
processor.do_resize = True  # Ensure processor resizes
processor.do_rescale = True  # Scale pixel values to [0,1]
processor.do_normalize = True  # Normalize with ImageNet stats


train_transform = A.Compose([
    # Geometric (preserving identity)
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.Affine(scale=0.85, rotate=20, p=0.5, shear=0.05),
    
    # Scale variations
    A.RandomSizedBBoxSafeCrop(height=640, width=640, erosion_rate=0.1, p=0.3),
    
    # Photometric (moderate)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=20, p=0.4),
    A.RandomGamma(gamma_limit=(90, 110), p=0.3),
    
    # Realistic effects
    A.RandomShadow(p=0.2),
    A.RandomFog(p=0.1),
    
    # Occlusion handling
    A.CoarseDropout(num_holes_range=(1,2), hole_height_range=(8,40), hole_width_range=(8,40), fill=0, p=0.2),
    
    # Noise (small amount for robustness)
    A.GaussNoise(p=0.1),

    # Resize to the same size as the YOLO model works
    A.Resize(height=640, width=640, p=1),
], bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"], min_visibility=0.3, min_area=25),
)

val_transform = A.Compose(
    [A.Resize(height=640, width=640, p=1),],
    bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
)

class VehicleLP(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(csv_file) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.labels = coco["annotations"]
        self.image_id_to_annotations = {}

        for label in self.labels:
                image_id = label["image_id"]
            
                if image_id not in self.image_id_to_annotations:
                    self.image_id_to_annotations[image_id] = []
    
                self.image_id_to_annotations[image_id].append(label)

    
    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        image_id = image_info["id"]

        annotations = self.image_id_to_annotations.get(image_id, [])

        # Extract bboxes and category IDs for Albumentations
        bboxes = [ann["bbox"] for ann in annotations]
        category_ids = [ann["category_id"] for ann in annotations]


        if self.transform and len(bboxes) > 0:
            # Albumentations expects a numpy array
            image_np = np.array(image)
            augmented = self.transform(image=image_np, bboxes=bboxes, category_ids=category_ids)
            
            # Reconstruct PIL image and filtered annotations
            image = Image.fromarray(augmented["image"])
            
            # Ensure any bounding boxes that were thrown out by rotation/cropping are omitted
            new_annotations = []
            for i, bbox in enumerate(augmented["bboxes"]):
                new_annotations.append({
                    "image_id": image_id,
                    "category_id": augmented["category_ids"][i],
                    "bbox": list(bbox),
                    "area": bbox[2] * bbox[3], # width * height
                    "iscrowd": 0
                })
            annotations = new_annotations

        return {
            "image": image,
            "annotations": {"image_id": image_id, "annotations": annotations}
        }

train_dataset = VehicleLP(
    root_dir='./datasets/bdd100k_yolo/train/images',
    csv_file='./datasets/bdd100k_yolo/train/train.json',
    transform=train_transform
)

val_dataset = VehicleLP(
    root_dir='./datasets/bdd100k_yolo/val/images',
    csv_file='./datasets/bdd100k_yolo/val/val.json',
    transform=val_transform
)

def collate_fn(batch):
    images = [item["image"] for item in batch]
    annotations = [item["annotations"] for item in batch]
    image_ids = [item["annotations"]["image_id"] for item in batch]
    
    inputs = processor(images=images, annotations=annotations, return_tensors="pt")
    inputs["image_id"] = torch.tensor(image_ids)
    
    return inputs


training_args = TrainingArguments(
    output_dir="./output/rtdetr",

    # Training Duration and Batch Size
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    dataloader_num_workers=4,
    num_train_epochs=30,
    #max_steps=300,

    # Learning Rate & Scheduler
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=5,

    # Optimizer
    optim="adamw_torch",
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999, 

    # Mixed Precision Training
    fp16=True, # Do not change

    save_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    remove_unused_columns=False,
    

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    
    
    logging_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)

trainer.train()
