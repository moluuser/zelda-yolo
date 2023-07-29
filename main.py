from ultralytics import YOLO
from utils.window import show_window

WINDOWS_OWNER = "Ryujinx"
BEST_PATH = "yolo/detect/train/weights/best.pt"

model = YOLO(BEST_PATH)

if __name__ == "__main__":
    # Must call `cv2.imshow` in main thread
    show_window(WINDOWS_OWNER, model)
