from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="coco128.yaml", epochs=10, imgsz=640, batch=16)
