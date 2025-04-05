#!/bin/bash

# Cho phép container truy cập X server của host
xhost +local:docker

# Chạy container và thực thi lệnh cài NumPy bên trong
sudo docker run -it \
    --ipc=host \
    --runtime=nvidia \
    --gpus all \
    -v /home/ivsr/Downloads/TensorrtNLG:/app \
    --device=/dev/video0:/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --net=host \
    ultralytics/ultralytics:latest-jetson-jetpack5 \
    /bin/bash -c "pip install numpy==1.23.5 && python3 /app/src/test.py"