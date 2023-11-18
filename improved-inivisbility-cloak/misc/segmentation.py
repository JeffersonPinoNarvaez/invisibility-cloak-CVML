import cv2
from ultralytics import YOLO

# Leer modelo
model = YOLO("improved-inivisbility-cloak/models/20231117_best.pt")      

# Abrir camara de video, el indice depende de cada equipo de computo
videoInput = cv2.VideoCapture(0)

while True:
    # Lee nuestros fotogramas
    ret, frame = videoInput.read()

    # Leemos los resultados para cada fotograma y su preduccion
    predicctions = model.predict(frame, imgsz = 640, conf = 0.78)

    # Visualizar  resultados
    labels = predicctions[0].plot()
    for predicction in predicctions:
        boxes = predicction.boxes  # Boxes object for bbox outputs
        masks = predicction.masks  # Masks object for segmentation masks outputs
        
    # Mostramos nuestros fotogramas
    cv2.imshow("Segmentation", labels)
    
    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break        

videoInput.release()
cv2.destroyAllWindows()