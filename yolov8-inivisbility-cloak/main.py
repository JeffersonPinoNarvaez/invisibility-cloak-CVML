import cv2
from ultralytics import YOLO
import numpy as np
import os

# Función para cargar archivos con verificación
def cargar_archivo(ruta, mensaje_error):
    if os.path.exists(ruta):
        return cv2.resize(cv2.imread(ruta), (capHeight, capWidth))
    else:
        print(f"Error: {mensaje_error} '{ruta}' no existe.")
        return None

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()
else:
    capWidth, capHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

background = cargar_archivo('yolov8-inivisbility-cloak/assets/background.jpg', 'El archivo de fondo')
model = YOLO('yolov8-inivisbility-cloak/models/20231117_best.pt') if os.path.exists('yolov8-inivisbility-cloak/models/20231117_best.pt') else None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resultados = model.predict(frame, imgsz=capWidth, conf=0.78) if model else None
    masks = resultados[0].masks if resultados else None
    poligono = np.zeros((capWidth, capHeight), dtype="uint8") if masks is None else (masks.data[0].cpu().numpy().astype("uint8")*255)
    poligono = cv2.resize(poligono, (capHeight, capWidth))
    background_masked = cv2.bitwise_and(background, background, mask=poligono) if background is not None else None
    frame = cv2.resize(frame, (capHeight, capWidth))
    final_result = cv2.add(frame, background_masked) if background_masked is not None else frame

    cv2.imshow('Invisibility Cloak', final_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()