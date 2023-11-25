from matplotlib import pyplot as plt
from ultralytics import YOLO
import numpy as np
import cv2
import os

def check_file_path(file_path):
    if os.path.exists(file_path):
        return file_path
    else:
        print(f"Error: the file '{file_path}' does not exist.")
        exit()

def model_prediction(model, img):
    model = YOLO(model)
    return model.predict(img, imgsz=640, conf=0.78)

def plot_data(data):
    plt.imshow(data, cmap='gray')
    plt.show()

def load_model_masks(data):
    top_masks = data[0].masks
    filtered_data = np.uint8(top_masks.data[0].cpu().numpy())
    plot_data(filtered_data)
    return filtered_data

def process_img(img_path, masking):
    image = cv2.imread(img_path)
    img_resize = cv2.resize(image, (640, 480))
    mask_formatted = cv2.bitwise_and(img_resize, img_resize, mask=masking)
    plot_data(cv2.cvtColor(mask_formatted, cv2.COLOR_BGR2RGB))

model = check_file_path('yolov8-inivisbility-cloak/models/20231117_best.pt')
red_cloak = check_file_path('yolov8-inivisbility-cloak/assets/red_cloak_testing.jpg')
background = check_file_path('yolov8-inivisbility-cloak/assets/background.jpg')

predictions = model_prediction(model, red_cloak)
masks = load_model_masks(predictions)
process_img(background, masks)