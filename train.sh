DATASET_PATH=''

yolo task=detect \
mode=train \
model=yolov8s.pt \
data=$DATASET_PATH/data.yaml \
epochs=100