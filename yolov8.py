from ultralytics import YOLO

#model = YOLO('../../../../../pretrained_models/yolov8n.pt')
#model = YOLO('../../../../../pretrained_models/yolov8s.pt')
model = YOLO('../../../../../pretrained_models/yolov8m.pt')
#model = YOLO('../../../../../pretrained_models/yolov8l.pt')
#model = YOLO('../../../../../pretrained_models/yolov8x.pt')
#model = YOLO('yolov8n.pt')

#results = model.track(source=0, show=True, tracker="botsort.yaml")
results = model.track(source=0, show=True, tracker="bytetrack.yaml")