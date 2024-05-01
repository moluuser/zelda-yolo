BEST_PATH='detect/train/weights/best.pt'
SOURCE='image.png'

yolo predict \
model=yolov8s.pt \
model=$BEST_PATH \
source=$SOURCE