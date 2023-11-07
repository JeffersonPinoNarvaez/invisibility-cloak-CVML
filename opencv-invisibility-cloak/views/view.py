import cv2

class VideoView:
    def show_instructions(self):
        print("Presiona la tecla 'q' para detener la captura de video.")
        cv2.waitKey(3000)  # Muestra las instrucciones durante 3 segundos

    def display_frame(self, frame):
        cv2.imshow("magic", frame)
        key = cv2.waitKey(1)

        # Si presionas la tecla 'q', detiene la captura de video
        if key & 0xFF == ord('q'):
            return False

        return True

    def release_output(self):
        cv2.destroyAllWindows()