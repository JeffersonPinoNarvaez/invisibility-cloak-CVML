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

def model_loading(model_path):
    return YOLO(model_path)

def model_predictions(model, img_path):
    img = cv2.imread(img_path)
    return model.predict(img, save=True, imgsz=640, conf=0.75)

def show_mask(mask):
    mask_data = mask.data[0].cpu().numpy()
    plt.imshow(mask_data, cmap='gray')
    plt.show()
    return np.uint8(mask_data)

def convert_to_uint8(data):
    return np.uint8(data)

def resize_and_apply_mask(original, mask, target_size=(640, 480)):
    img_resize = cv2.resize(original, target_size)
    mask_formatted = cv2.bitwise_and(img_resize, img_resize, mask=mask)
    plt.imshow(cv2.cvtColor(mask_formatted, cv2.COLOR_BGR2RGB))
    plt.show()

def main():

    model_path = check_file_path("../models/20231117_best.pt")
    model = model_loading(model_path)

    image_path = '../assets/red_cloak.jpg'
    results = model_predictions(model, image_path)
    result = results[0]
    masks = result.masks

    first_mask = masks[0]
    first_mask_data = first_mask.data[0].cpu().numpy()
    show_mask(first_mask)

    first_mask_formatted = convert_to_uint8(first_mask_data)

    original = cv2.imread(image_path)
    resize_and_apply_mask(original, first_mask_formatted)

if __name__ == "__main__":
    main()