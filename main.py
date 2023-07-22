import time
import cv2
import mss
import numpy as np
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
from colorama import Fore, Style
from ultralytics import YOLO
import pyautogui
import pygetwindow as gw

WINDOWS_TITLE = "Ryujinx  1.1.0-macos1 - 塞尔达传说 王国之泪 v1.0.0 (0100F2C0115B6000) (64-bit)"
BEST_PATH = "./detect/train/weights/best.pt"

SIGHTING_Y_OFFSET = 30
SIGHTING_IS_VISIBLE = True

MODEL_THRESHOLD = 0.7

model = YOLO(BEST_PATH)


def get_window_geometry(window_title):
    # Get all windows on the screen
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)

    print(Fore.GREEN + "Windows list:" + Style.RESET_ALL)
    for window_info in window_list:
        name = window_info.get("kCGWindowName")
        owner_name = window_info.get("kCGWindowOwnerName")
        print("{} - {})".format(name, owner_name))
        if window_info.get("kCGWindowName", "") == window_title:
            # Get the window size and position
            top = window_info["kCGWindowBounds"]["Y"]
            left = window_info["kCGWindowBounds"]["X"]
            width = window_info["kCGWindowBounds"]["Width"]
            height = window_info["kCGWindowBounds"]["Height"]
            return {"top": top, "left": left, "width": width, "height": height}

    return None


def show_window(window_title):
    # Find the window
    window_geometry = get_window_geometry(window_title)
    if window_geometry is None:
        print(Fore.RED + "Cannot find window with title '{}'.".format(window_title) + Style.RESET_ALL)
        return

    with mss.mss() as sct:
        while True:
            # Get screenshot
            img = np.array(sct.grab(window_geometry))

            # BGRA to BGR
            img_brg = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            res = model.predict(source=img_brg, conf=MODEL_THRESHOLD)
            r = res[0]

            res_plotted = r.plot()

            if SIGHTING_IS_VISIBLE:
                image_height, image_width, _ = res_plotted.shape
                center_x = int(image_width // 2)
                center_y = int(image_height // 2 + SIGHTING_Y_OFFSET)
                cv2.circle(res_plotted, (center_x, center_y), 5, (0, 0, 255), -1)

            highest = None

            if len(r.boxes) > 0:
                # Find the highest threshold box
                for box in r.boxes:
                    if highest is None or box.conf > highest.conf:
                        highest = box

            if highest is not None:
                arr = highest.xywh.tolist()[0]
                x = arr[0]
                y = arr[1]
                w = arr[2]
                h = arr[3]
                # print("x: {}, y: {}, w: {}, h: {}".format(x, y, w, h))
                center_x = int(x)
                center_y = int(y)
                print("center_x: {}, center_y: {}".format(center_x, center_y))
                cv2.circle(res_plotted, (center_x, center_y), 5, (0, 255, 0), -1)

            # pyautogui.moveTo(center_x, center_y)
            # pyautogui.click()
            # time.sleep(0.1)

            # Display the picture
            resized_img = resize_image(res_plotted, 0)
            title = get_first_word_before_space(window_title).encode("gbk").decode(errors="ignore")
            cv2.imshow(title, resized_img)

            # Press "q" to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Close the window
    cv2.destroyAllWindows()


def get_first_word_before_space(text):
    words = text.split(' ')
    if len(words) > 1:
        return words[0]
    else:
        return text


def resize_image(img, target_width):
    if target_width == 0:
        return img
    height, width = img.shape[:2]
    target_height = int(target_width * height / width)
    resized_img = cv2.resize(img, (target_width, target_height))
    return resized_img


def control_link():
    # If in zelda window
    active_window = gw.getActiveWindow()
    if WINDOWS_TITLE in active_window:
        # In zelda window
        pyautogui.keyDown('w')
        time.sleep(2)
        pyautogui.keyUp('w')


if __name__ == "__main__":
    show_window(WINDOWS_TITLE)
    # control_link()
