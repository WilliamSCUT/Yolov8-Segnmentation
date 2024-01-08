from ultralytics import YOLO
'''
model = YOLO('yolov8s.pt')
results = model.train(data='./new_data/foot.yaml', epochs=100, imgsz=640)
'''
model = YOLO('yolov8s-seg.pt')
results = model.train(data='./gear_classfication/gear_400.yaml', epochs=200, imgsz=640)