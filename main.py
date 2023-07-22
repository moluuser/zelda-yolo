import cv2
import mss
import numpy as np
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
from colorama import Fore, Style
from ultralytics import YOLO

WINDOWS_TITLE = "Ryujinx  1.1.0-macos1 - 塞尔达传说 王国之泪 v1.0.0 (0100F2C0115B6000) (64-bit)"
BEST_PATH = "/Users/chenyang/Developer/PycharmProjects/zelda-yolo/detect/train/weights/best.pt"

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

            res = model.predict(source=img_brg)
            res_plotted = res[0].plot()

            # Display the picture
            cv2.imshow(get_first_word_before_space(window_title).encode("gbk").decode(errors="ignore"), res_plotted)

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


if __name__ == "__main__":
    window_title = WINDOWS_TITLE
    show_window(window_title)
