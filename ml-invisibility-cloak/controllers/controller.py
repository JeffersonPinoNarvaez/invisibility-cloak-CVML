import cv2
import time

from models.model import VideoModel
from views.view import VideoView

class VideoController:
    def __init__(self):
        self.video_model = VideoModel()
        self.video_view = VideoView()

    def start_processing(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("No se pudo abrir la cámara")

        time.sleep(3)

        self.video_model.capture_background(cap)
        self.video_view.show_instructions()

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            processed_frame = self.video_model.process_frame(img)
            self.video_view.display_frame(processed_frame)

        cap.release()
        self.video_view.release_output()