import time
import cv2
import mss
import numpy as np
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
from colorama import Fore, Style
from ultralytics import YOLO
import pyautogui
import pygetwindow as gw
from queue import Queue
from threading import Thread

WINDOWS_TITLE = "Ryujinx  1.1.0-macos1 - 塞尔达传说 王国之泪 v1.0.0 (0100F2C0115B6000) (64-bit)"
BEST_PATH = "./detect/train/weights/best.pt"
SIGHTING_Y_OFFSET = 30
SIGHTING_IS_VISIBLE = True
MODEL_THRESHOLD = 0.7
SHOOTING_THRESHOLD = 130
PRESS_RATIO = 0.02

center_x = 0
center_y = 0

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


def show_window(window_title, out_q):
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
                global center_x, center_y
                center_x = int(image_width // 2)
                center_y = int(image_height // 2 + SIGHTING_Y_OFFSET)
                # print(center_x, center_y)
                cv2.circle(res_plotted, (center_x, center_y), 5, (0, 0, 255), -1)

            highest = None

            if len(r.boxes) > 0:
                # Run alay...
                if len(r.boxes) > 3:
                    # TODO
                    pass

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
                cv2.circle(res_plotted, (int(x), int(y)), 5, (0, 255, 0), -1)

                out_q.put(arr)

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


def control_link(in_q, out_x_q, out_y_q):
    while True:
        active_window = gw.getActiveWindow()
        if WINDOWS_TITLE in active_window:
            # In zelda window
            arr = in_q.get()
            x = int(arr[0])
            y = int(arr[1])
            w = int(arr[2])
            h = int(arr[3])
            # print("x: {}, y: {}, w: {}, h: {}".format(x, y, w, h))

            move_to_center(out_x_q, out_y_q, x, y)

            if w > SHOOTING_THRESHOLD or h > SHOOTING_THRESHOLD:
                # Close-up attack
                press_key('v')
            elif abs(x - center_x) < 10 and abs(y - center_y) < 10:
                # Shoot
                press_key('o')

            time.sleep(0.5)


def move_to_center(out_x_q, out_y_q, x, y):
    print("x: {}, y: {}".format(abs(x - center_x), abs(y - center_y)))
    if x < center_x:
        out_x_q.put(['a', abs(x - center_x) * PRESS_RATIO])
    elif x > center_x:
        out_x_q.put(['d', abs(x - center_x) * PRESS_RATIO])

    if y < center_y:
        out_y_q.put(['w', abs(y - center_y) * PRESS_RATIO])
    elif y > center_y:
        out_y_q.put(['s', abs(y - center_y) * PRESS_RATIO])


def press_key(key, duration=0.1):
    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)


def press_x(in_q):
    while True:
        arr = in_q.get()
        key = arr[0]
        duration = arr[1]
        press_key(key, duration)


def press_y(in_q):
    while True:
        arr = in_q.get()
        key = arr[0]
        duration = arr[1]
        press_key(key, duration)


if __name__ == "__main__":
    q = Queue()
    x_q = Queue()
    y_q = Queue()

    thread_control = Thread(target=control_link, args=(q, x_q, y_q))
    thread_control.start()

    thread_press_x = Thread(target=press_x, args=(x_q,))
    thread_press_x.start()

    thread_press_y = Thread(target=press_y, args=(y_q,))
    thread_press_y.start()

    # Must call `cv2.imshow` in main thread
    show_window(WINDOWS_TITLE, q)
