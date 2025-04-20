YOLO Object Detection Project on Jetson with Docker
This project demonstrates how to use the YOLO model (via the ultralytics library) for object detection on a Jetson device using a Docker container. It supports detection on images, videos, and real-time webcam input, with results saved to a specified directory. The project leverages TensorRT for optimized inference on NVIDIA GPUs.
Prerequisites

Hardware: NVIDIA Jetson device (e.g., Jetson Xavier) with a webcam (optional for real-time detection).
Software:
Docker installed on the host.
NVIDIA Container Toolkit for GPU support.


Files:
YOLO model: yolo11n.pt or yolo11n.engine (TensorRT format).
Input data: Images, videos, or webcam.



Setup
1. Pull the Docker Image
The project uses the official ultralytics image for Jetson with JetPack 5:
t=ultralytics/ultralytics:latest-jetson-jetpack5
sudo docker pull $t

2. Run the Docker Container
Run the container with GPU support, webcam access, and bind mounts for input/output:
sudo docker run -it --ipc=host --runtime=nvidia \
 --device=/dev/video0:/dev/video0 \
 -v /path/to/your/data:/host \
 -v /path/to/your/output:/ultralytics/output \
 $t


--device=/dev/video0:/dev/video0: Mount webcam (adjust /dev/video0 if needed).
-v /path/to/your/data:/host: Mount directory containing input data and scripts.
-v /path/to/your/output:/ultralytics/output: Mount directory to save detection results.

Example (replace /home/ivsr/Downloads/TensorrtNLG with your actual path):
sudo docker run -it --ipc=host --runtime=nvidia \
 --device=/dev/video0:/dev/video0 \
 -v /home/ivsr/Downloads/TensorrtNLG:/host \
 -v /home/ivsr/Downloads/TensorrtNLG/output:/ultralytics/output \
 $t

3. Install Dependencies (Inside Container)
Ensure required libraries are installed:
pip3 install ultralytics opencv-python

How to Run
Options:

Detection stream webcam - YOLO (default)
Detection video local
Detection images local/URL

Option 1: Detection Stream Webcam - YOLO

Create a script detection_cam.sh to run the container and perform detection with webcam:
nano /path/to/your/data/detection_cam.sh

Add the following content:
#!/bin/bash

# Allow container to access the host's X server
xhost +local:docker

# Run container and execute detection script
sudo docker run -it \
    --ipc=host \
    --runtime=nvidia \
    --gpus all \
    -v /path/to/your/data:/app \
    --device=/dev/video0:/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --net=host \
    ultralytics/ultralytics:latest-jetson-jetpack5 \
    /bin/bash -c "pip install numpy==1.23.5 && python3 /app/src/test.py"

Example (replace /path/to/your/data with your actual path):
#!/bin/bash

xhost +local:docker

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

Save and exit (Ctrl+O, Enter, Ctrl+X).

Make the script executable and run:
cd /path/to/your/data
chmod +x detection_cam.sh
./detection_cam.sh


The script will run the container, install numpy==1.23.5, and execute /app/src/test.py (mapped to /path/to/your/data/src/test.py on host).
Ensure test.py is configured to save results to /ultralytics/output/webcam_results (mapped to /path/to/your/output/webcam_results on host).



Option 2: Detection Video Local

Modify detection_cam.sh to run detection on a local video:
nano /path/to/your/data/detection_cam.sh

Update the content of /app/src/test.py to process a video file (example /app/data/your_video.mp4), or create a new script to run via CLI:
#!/bin/bash

xhost +local:docker

sudo docker run -it \
    --ipc=host \
    --runtime=nvidia \
    --gpus all \
    -v /path/to/your/data:/app \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --net=host \
    ultralytics/ultralytics:latest-jetson-jetpack5 \
    /bin/bash -c "yolo predict model=/ultralytics/yolo11n.engine source=/app/data/your_video.mp4 project=/ultralytics/output name=video_results"

Save and exit.

Run the script:
cd /path/to/your/data
chmod +x detection_cam.sh
./detection_cam.sh


Replace /app/data/your_video.mp4 with the path to your video file.
Results will be saved to /ultralytics/output/video_results.



Option 3: Detection Images Local/URL

Modify detection_cam.sh for image detection:
nano /path/to/your/data/detection_cam.sh

Update the content:
#!/bin/bash

xhost +local:docker

sudo docker run -it \
    --ipc=host \
    --runtime=nvidia \
    --gpus all \
    -v /path/to/your/data:/app \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --net=host \
    ultralytics/ultralytics:latest-jetson-jetpack5 \
    /bin/bash -c "yolo predict model=/ultralytics/yolo11n.engine source=/app/data/pexels-joshsorenson-139303.jpg project=/ultralytics/output name=image_results"

Or for a URL:
#!/bin/bash

xhost +local:docker

sudo docker run -it \
    --ipc=host \
    --runtime=nvidia \
    --gpus all \
    -v /path/to/your/data:/app \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --net=host \
    ultralytics/ultralytics:latest-jetson-jetpack5 \
    /bin/bash -c "yolo predict model=/ultralytics/yolo11n.engine source='https://ultralytics.com/images/bus.jpg' project=/ultralytics/output name=image_results"

Save and exit.

Run the script:
cd /path/to/your/data
chmod +x detection_cam.sh
./detection_cam.sh


Results will be saved to /ultralytics/output/image_results.



Add to Path for One-Command Run: "docker_trt_yolo"
Make the script globally accessible on the host:
sudo cp /path/to/your/data/detection_cam.sh /usr/local/bin/docker_trt_yolo
chmod +x /usr/local/bin/docker_trt_yolo
docker_trt_yolo


Now you can run docker_trt_yolo from any directory on the host to execute the detection script.

Troubleshooting
1. ImportError: No module named ultralytics

Cause: The /ultralytics/ultralytics/ directory causes import conflicts.

Solution: Use PYTHONPATH to prioritize the pip-installed module:
PYTHONPATH=/usr/local/lib/python3.8/dist-packages python3 your_script.py



2. FileNotFoundError: No such file or directory: 'yolov11n.pt'

Cause: Incorrect model path or typo.

Solution: Use the correct path:
model = YOLO("/ultralytics/yolo11n.pt")  # or /ultralytics/yolo11n.engine



3. qt.qpa.xcb: could not connect to display

Cause: Container lacks GUI support.
Solution:
Disable GUI display (remove cv2.imshow or results.show()).
Or ensure X11 forwarding is enabled (already included in detection_cam.sh).



4. Webcam Not Working

Cause: Webcam not mounted or not accessible.
Solution:
Check webcam device:
ls /dev/video*


Ensure container is run with --device=/dev/video0.




Directory Structure

/host: Directory mounted from host (e.g., /home/ivsr/Downloads/TensorrtNLG).
Contains scripts (detection_cam.sh, src/test.py).
Contains input data (/host/data/pexels-joshsorenson-139303.jpg).


/ultralytics: Directory containing model files (yolo11n.pt, yolo11n.engine).
/ultralytics/output: Directory to save detection results (mapped to host).

License
This project uses the ultralytics library under the AGPL-3.0 license. See Ultralytics License for details.
