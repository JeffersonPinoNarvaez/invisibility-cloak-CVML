import cv2
import numpy as np

class VideoModel:
    def __init__(self):
        self.background = None

    def capture_background(self, cap):
        background = 0
        for i in range(60):
            ret, background = cap.read()
        self.background = background

    def process_frame(self, frame):
        frame = np.flip(frame, axis=1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Generar máscaras para detectar un color específico (en este caso, rojo)
        lower_red = np.array([0, 120, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask1 = mask1 + mask2

        # Aplicar operaciones morfológicas a la máscara
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # Crear una máscara invertida para segmentar el color rojo del fotograma
        mask2 = cv2.bitwise_not(mask1)

        # Segmentar la parte roja del fotograma
        res1 = cv2.bitwise_and(frame, frame, mask=mask2)

        # Crear una imagen que muestre el fondo estático solo para la región enmascarada
        res2 = cv2.bitwise_and(self.background, self.background, mask=mask1)

        # Generar la salida final
        final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
        return final_output