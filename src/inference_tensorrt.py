# -*- coding: utf-8 -*-


from ultralytics import YOLO

 
trt_model = YOLO("yolo11n.engine")

results = trt_model("/host/27260-362770008_small.mp4",save = True)