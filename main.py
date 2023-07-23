import time
import cv2
import mss
import numpy as np
import pynput
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
from colorama import Fore, Style
from ultralytics import YOLO
import pygetwindow as gw
from queue import Queue
from threading import Thread

WINDOWS_OWNER = "Ryujinx"
BEST_PATH = "./detect/train/weights/best.pt"
SIGHTING_Y_OFFSET = 30
SIGHTING_IS_VISIBLE = True
MODEL_THRESHOLD = 0.6
SHOOTING_THRESHOLD = 130
PRESS_RATIO = 0.005
ONCE_DURATION = 1
ONLY_SHOOT = True

center_x = 0
center_y = 0

model = YOLO(BEST_PATH)
ctr = pynput.keyboard.Controller()


def get_window_geometry(window_owner):
    # Get all windows on the screen
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)

    print(Fore.GREEN + "Windows list:" + Style.RESET_ALL)
    for window_info in window_list:
        name = window_info.get("kCGWindowName")
        owner_name = window_info.get("kCGWindowOwnerName")
        print("{} - {})".format(name, owner_name))
        if owner_name == window_owner:
            # Get the window size and position
            top = window_info["kCGWindowBounds"]["Y"]
            left = window_info["kCGWindowBounds"]["X"]
            width = window_info["kCGWindowBounds"]["Width"]
            height = window_info["kCGWindowBounds"]["Height"]
            return {"top": top, "left": left, "width": width, "height": height}

    return None


def show_window(window_owner, out_q):
    # Find the window
    window_geometry = get_window_geometry(window_owner)
    if window_geometry is None:
        print(Fore.RED + "Cannot find window with owner '{}'.".format(window_owner) + Style.RESET_ALL)
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
                # Run away...
                if len(r.boxes) > 3:
                    # TODO
                    pass

                # Find the highest threshold box
                for box in r.boxes:
                    if highest is None or box.conf > highest.conf:
                        highest = box

            if highest is not None:
                arr = highest.xywh.tolist()[0]
                x, y, w, h = arr[:4]
                # print("x: {}, y: {}, w: {}, h: {}".format(x, y, w, h))
                cv2.circle(res_plotted, (int(x), int(y)), 5, (0, 255, 0), -1)

                if out_q.qsize() > 3:
                    out_q.queue.clear()

                out_q.put(arr)
                print("Current move queue size: {}".format(out_q.qsize()))

            # Display the picture
            resized_img = resize_image(res_plotted, 0)
            cv2.imshow(window_owner, resized_img)

            # Press "q" to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Close the window
    cv2.destroyAllWindows()


def resize_image(img, target_width):
    if target_width == 0:
        return img
    height, width = img.shape[:2]
    target_height = int(target_width * height / width)
    resized_img = cv2.resize(img, (target_width, target_height))
    return resized_img


def control_link(in_q, out_key_q):
    while True:
        active_window = gw.getActiveWindow()
        if WINDOWS_OWNER in active_window:
            # In zelda window
            arr = in_q.get()
            x, y, w, h = arr[:4]
            # print("x: {}, y: {}, w: {}, h: {}".format(x, y, w, h))

            # move_to_center_by_moving(out_key_q, x, y)

            if abs(x - center_x) < 500 and abs(y - center_y) < 500:
                if not ONLY_SHOOT and (w > SHOOTING_THRESHOLD or h > SHOOTING_THRESHOLD):
                    # Close-up attack
                    press_key('v')

            if ONLY_SHOOT:
                # Shoot
                ctr.press('o')
                move_to_center_by_view(x, y)
                ctr.release('o')

            time.sleep(ONCE_DURATION)


def move_to_center_by_view(x, y):
    x_ratio = 0.0005
    y_ratio = 0.001
    x_list = []
    if x < center_x:
        x_list.append('j')
    elif x > center_x:
        x_list.append('l')
    x_list.append(abs(x - center_x) * x_ratio)
    ctr.press(x_list[0])
    time.sleep(x_list[1])
    ctr.release(x_list[0])

    y_list = []
    if y < center_y:
        y_list.append('i')
    elif y > center_y:
        y_list.append('k')
    y_list.append(abs(y - center_y) * y_ratio)
    ctr.press(y_list[0])
    time.sleep(y_list[1])
    ctr.release(y_list[0])


def move_to_center_by_moving(out_key_q, x, y):
    # print("x_offset: {}, y_offset: {}".format(abs(x - center_x), abs(y - center_y)))
    key_list = []
    if x < center_x:
        key_list.append('a')
    elif x > center_x:
        key_list.append('d')

    if y < center_y:
        key_list.append('w')
    elif y > center_y:
        key_list.append('s')

    if len(key_list) > 0:
        max_offset = max(abs(x - center_x), abs(y - center_y))
        key_list.append(max_offset * PRESS_RATIO)

        # `key_list` will be like ['a', 'w', 5]
        out_key_q.put(key_list)


def press_key(key, duration=0.1):
    if duration < 0.1:
        duration = 0.1
    if duration > ONCE_DURATION:
        duration = ONCE_DURATION

    print(Fore.GREEN + "Pressing {} for {} seconds.".format(key, str(duration)) + Style.RESET_ALL)

    for k in key:
        if k == '':
            continue
        ctr.press(k)

    time.sleep(duration)

    for k in key:
        if k == '':
            continue
        ctr.release(k)

    time.sleep(0.1)


def control_press_key(in_q):
    while True:
        arr = in_q.get()
        key = arr[:2]
        duration = arr[2]
        press_key(key, duration)


if __name__ == "__main__":
    q = Queue()
    key_q = Queue()

    thread_control_link = Thread(target=control_link, args=(q, key_q))
    thread_control_link.start()

    thread_control_press_key = Thread(target=control_press_key, args=(key_q,))
    thread_control_press_key.start()

    # Must call `cv2.imshow` in main thread
    show_window(WINDOWS_OWNER, q)
