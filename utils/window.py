import cv2
import mss
import numpy as np
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
from colorama import Fore, Style

from config.const import SIGHTING_IS_VISIBLE, MODEL_THRESHOLD, SIGHTING_Y_OFFSET
from utils.image import resize_image


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


def show_window(window_owner, model):
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

            # Draw the center
            if SIGHTING_IS_VISIBLE:
                image_height, image_width, _ = res_plotted.shape
                center_x = int(image_width // 2)
                center_y = int(image_height // 2 + SIGHTING_Y_OFFSET)
                # print(center_x, center_y)
                cv2.circle(res_plotted, (center_x, center_y), 5, (0, 0, 255), -1)

            highest = None

            if len(r.boxes) > 0:
                # Find the highest threshold box
                for box in r.boxes:
                    if highest is None or box.conf > highest.conf:
                        highest = box

            if highest is not None:
                arr = highest.xywh.tolist()[0]
                x, y, w, h = arr[:4]
                # print("x: {}, y: {}, w: {}, h: {}".format(x, y, w, h))
                # Draw the center on the highest threshold box
                cv2.circle(res_plotted, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Display the picture
            resized_img = resize_image(res_plotted, 0)
            cv2.imshow(window_owner, resized_img)

            # Press "q" to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Close the window
    cv2.destroyAllWindows()
