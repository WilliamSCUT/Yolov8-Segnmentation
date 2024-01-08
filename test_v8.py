from ultralytics import YOLO

# 加载预训练的YOLOv8n模型
model_path = 'model/four_object_segnmentation_model/weights/best.pt'
model = YOLO(model_path)

# 在'bus.jpg'上运行推理，并附加参数
results = model.predict('bus.jpg', save=False, conf=0.5)

for result in results:
    boxes = result.boxes  # 边界框输出的 Boxes 对象
    masks = result.masks  # 分割掩码输出的 Masks 对象
    keypoints = result.keypoints  # 姿态输出的 Keypoints 对象
    probs = result.probs  # 分类输出的 Probs 对象