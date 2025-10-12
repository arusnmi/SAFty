from ultralytics import YOLO

model=YOLO("yolo11n.pt")

model.train(data="dataset_custom.yaml", epochs=100, imgsz=640,  batch=8, device=0, workers=0)
 