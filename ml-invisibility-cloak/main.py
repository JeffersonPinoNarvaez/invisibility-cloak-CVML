import cv2
import time
from controllers.controller import VideoController

def main():
    video_controller = VideoController()
    video_controller.start_processing()

if __name__ == "__main__":
    main()