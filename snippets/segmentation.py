import cv2
from ultralytics import YOLO
import os

def check_file_path(file_path):
    if os.path.exists(file_path):
        return file_path
    else:
        print(f"Error: the file '{file_path}' does not exist.")
        exit()

def model_loading(model_path):
    return YOLO(model_path)

def model_predictions(model, img):
    return model.predict(img, imgsz=640, conf=0.78)

model_path = check_file_path("../models/20231117_best.pt")      
model = model_loading(model_path)

video_input = cv2.VideoCapture(0)

while True:
    ret, frame = video_input.read()
    predictions = model_predictions(model, frame)

    labels = predictions[0].plot()
    for prediction in predictions:
        boxes = prediction.boxes
        masks = prediction.masks
    
    cv2.imshow("Segmentation", labels)
    
    if cv2.waitKey(1) == 27:
        break        

video_input.release()
cv2.destroyAllWindows()