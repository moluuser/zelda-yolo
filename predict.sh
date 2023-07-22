BEST_PATH=''
SOURCE=''

yolo predict \
model=yolov8s.pt \
model=$BEST_PATH/weights/best.pt \
source=$SOURCE