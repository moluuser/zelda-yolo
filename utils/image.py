import cv2


def resize_image(img, target_width):
    if target_width == 0:
        return img
    height, width = img.shape[:2]
    target_height = int(target_width * height / width)
    resized_img = cv2.resize(img, (target_width, target_height))
    return resized_img
