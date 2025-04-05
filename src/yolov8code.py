from ultralytics import YOLO
#model = YOLO("yolov8n.pt")
#model.export(format="engine",device = '0')  # creates 'yolo11n.engine'
trt_model = YOLO("/app/yolov8n.engine")
results = trt_model("/app/data/pexels-joshsorenson-139303.jpg",save = True)