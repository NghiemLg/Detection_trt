#  <span style="color:Yellow" >How to run </span>
## _Option:_
- Detection stream webcam -yolov8 (default)
- Detection videolocal
- Detection images local/url

## Step1 :<span style ="color:green"> Clone github repo </span>
    git clone https://github.com/NghiemLg/Clasification_trt.git
### Option1 : <span style="color:green"> Detection stream webcam -yolov8 </span>
    cd /TensorrtNLG/script/detection_cam.sh
    chmod +x detection_cam.sh
    ./detection_cam.sh
### Option 2: <span style="color:green"> Ditection videolocal</span>
    cd script/detection_cam.sh
    chmod +x detection_cam.sh
    ./detection_cam.sh
### Add to path for  run with one comand: "_docker_trt_yolo_"
    sudo cp detection_cam.sh /usr/local/bin/docker_trt_yolo
    docker_trt_yolo 
