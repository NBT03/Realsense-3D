from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

# freeze_support()

# print("GPU Available: ", torch.cuda.is_available())
# print("GPU Name: ", torch.cuda.get_device_name(0))

# Load a COCO-pretrained YOLO11n model
model = YOLO("/home/batien/Desktop/Realsense-3D/best.pt")

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(
#     data="/home/pc/tienai/yolo/box_seg/shapes-2/data.yaml", 
#     epochs=300, 
#     imgsz=640,
#     device=0,
#     batch=16,
#     name="train",
#     patience=30,
#     mixup=0.1
#     )

# Evaluate model performance on the validation set
# metrics = model.val()

# # Perform object detection on an image
results = model("/home/batien/Desktop/Realsense-3D/data_box/IMG_20250330_103443.jpg")
results[0].show()