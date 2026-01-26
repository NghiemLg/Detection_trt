---

# YOLO Object Detection on NVIDIA Jetson using Docker & TensorRT

This project demonstrates how to deploy **YOLO object detection** on **NVIDIA Jetson devices** using **Docker** and the **Ultralytics YOLO framework**.
It supports inference on **images**, **videos**, and **real-time webcam streams**, with optional **TensorRT acceleration** for optimized performance on NVIDIA GPUs.

---

## Features

* Supports YOLO inference on:

  * Webcam (real-time)
  * Local videos
  * Local images or image URLs
* Docker-based deployment (clean and reproducible)
* Optimized inference using **TensorRT**
* Compatible with **JetPack 5**
* Output results automatically saved to host directory

---

## Prerequisites

### Hardware

* NVIDIA Jetson device (e.g., Xavier, Orin, Nano)
* Webcam (optional, for real-time detection)

### Software

* Docker
* NVIDIA Container Toolkit (for GPU support)
* JetPack 5.x

---

## Required Files

* **YOLO model**

  * `yolo11n.pt` (PyTorch)
  * or `yolo11n.engine` (TensorRT)
* **Input data**

  * Images, videos, or webcam
* **Scripts**

  * `detection_cam.sh`
  * `src/test.py`

---

## Setup

### 1. Pull Docker Image

The project uses the official Ultralytics Jetson image:

```bash
t=ultralytics/ultralytics:latest-jetson-jetpack5
sudo docker pull $t
```

---

### 2. Run Docker Container (Basic)

Run the container with GPU support, webcam access, and mounted directories:

```bash
sudo docker run -it --ipc=host --runtime=nvidia \
  --device=/dev/video0:/dev/video0 \
  -v /path/to/your/data:/host \
  -v /path/to/your/output:/ultralytics/output \
  $t
```

**Explanation**

* `--device=/dev/video0` : Webcam access
* `/host` : Input data & scripts
* `/ultralytics/output` : Detection results

**Example**

```bash
sudo docker run -it --ipc=host --runtime=nvidia \
  --device=/dev/video0:/dev/video0 \
  -v /home/ivsr/Downloads/TensorrtNLG:/host \
  -v /home/ivsr/Downloads/TensorrtNLG/output:/ultralytics/output \
  $t
```

---

### 3. Install Dependencies (Inside Container)

```bash
pip3 install ultralytics opencv-python
```

---

## How to Run Detection

### Supported Modes

1. Webcam (real-time)
2. Local video
3. Local image or image URL

---

## Option 1: Webcam Detection (Real-Time)

### Create Script

```bash
nano /path/to/your/data/detection_cam.sh
```

### Script Content

```bash
#!/bin/bash

xhost +local:docker

sudo docker run -it \
  --ipc=host \
  --runtime=nvidia \
  --gpus all \
  --device=/dev/video0:/dev/video0 \
  -v /path/to/your/data:/app \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --net=host \
  ultralytics/ultralytics:latest-jetson-jetpack5 \
  /bin/bash -c "pip install numpy==1.23.5 && python3 /app/src/test.py"
```

### Example

```bash
-v /home/ivsr/Downloads/TensorrtNLG:/app
```

### Run

```bash
chmod +x detection_cam.sh
./detection_cam.sh
```

**Note**

* Results should be saved to:

  ```
  /ultralytics/output/webcam_results
  ```

---

## Option 2: Local Video Detection

Update `detection_cam.sh`:

```bash
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
  /bin/bash -c "yolo predict model=/ultralytics/yolo11n.engine \
    source=/app/data/your_video.mp4 \
    project=/ultralytics/output name=video_results"
```

Results are saved to:

```
/ultralytics/output/video_results
```

---

## Option 3: Image Detection (Local or URL)

### Local Image

```bash
yolo predict model=/ultralytics/yolo11n.engine \
  source=/app/data/pexels-joshsorenson-139303.jpg \
  project=/ultralytics/output name=image_results
```

### Image URL

```bash
yolo predict model=/ultralytics/yolo11n.engine \
  source="https://ultralytics.com/images/bus.jpg" \
  project=/ultralytics/output name=image_results
```

---

## One-Command Execution (Optional)

Make the script globally accessible:

```bash
sudo cp detection_cam.sh /usr/local/bin/docker_trt_yolo
sudo chmod +x /usr/local/bin/docker_trt_yolo
```

Run from anywhere:

```bash
docker_trt_yolo
```

---

## Directory Structure

```
/host
 ├── detection_cam.sh
 ├── src/
 │   └── test.py
 └── data/
     └── input images/videos

/ultralytics
 ├── yolo11n.pt
 ├── yolo11n.engine
 └── output/
     ├── webcam_results
     ├── video_results
     └── image_results
```

---

## Troubleshooting

### 1. `ImportError: No module named ultralytics`

**Cause**
Conflict with `/ultralytics/ultralytics/` directory.

**Solution**

```bash
PYTHONPATH=/usr/local/lib/python3.8/dist-packages python3 your_script.py
```

---

### 2. `FileNotFoundError: yolov11n.pt`

**Cause**
Incorrect model path or typo.

**Solution**

```python
model = YOLO("/ultralytics/yolo11n.pt")
# or
model = YOLO("/ultralytics/yolo11n.engine")
```

---

### 3. `qt.qpa.xcb: could not connect to display`

**Cause**
X11 not properly forwarded.

**Solution**

* Disable GUI output (`cv2.imshow`, `results.show()`)
* Or ensure X11 forwarding (already configured in scripts)

---

### 4. Webcam Not Working

Check device:

```bash
ls /dev/video*
```

Ensure Docker is run with:

```bash
--device=/dev/video0:/dev/video0
```

---

## License

This project uses the **Ultralytics YOLO** library under the **AGPL-3.0 License**.
Refer to the Ultralytics license documentation for details.


