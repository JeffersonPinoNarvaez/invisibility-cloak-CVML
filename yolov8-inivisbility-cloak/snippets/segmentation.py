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

# Load the model
model_path = check_file_path("improved-inivisbility-cloak/models/20231117_best.pt")      
model = model_loading(model_path)

# Open video camera, the index depends on the computer's setup
video_input = cv2.VideoCapture(0)

while True:
    # Read frames
    ret, frame = video_input.read()

    # Get predictions for each frame
    predictions = model_predictions(model, frame)

    # Display results
    labels = predictions[0].plot()
    for prediction in predictions:
        boxes = prediction.boxes
        masks = prediction.masks
    
    # Show frames
    cv2.imshow("Segmentation", labels)
    
    # Exit the program
    if cv2.waitKey(1) == 27:
        break        

video_input.release()
cv2.destroyAllWindows()