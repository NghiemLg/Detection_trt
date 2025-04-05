from ultralytics import YOLO
import cv2
import time

# Load a YOLO11n PyTorch model
model = YOLO("yolo11n.pt")

# Export the model to TensorRT (chỉ chạy một lần, sau đó comment lại)

#model.export(format="engine")  # creates 'yolo11n.engine'

# Load the exported TensorRT model
trt_model = YOLO("/app/yolo11n.engine")

# Mở webcam
cap = cv2.VideoCapture(0)  # 0 tương ứng với /dev/video0 trong container
if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

# Biến để tính FPS
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ webcam")
        break

    # Thực hiện inference trên frame từ webcam
    results = trt_model(frame)

    # Vẽ kết quả (bounding box, label) lên frame
    annotated_frame = results[0].plot()

    # Tính FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # Vẽ FPS lên frame
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(
        annotated_frame, 
        fps_text, 
        (10, 30),  # Vị trí (x, y) trên frame
        cv2.FONT_HERSHEY_SIMPLEX, 
        1,  # Kích thước chữ
        (0, 255, 0),  # Màu chữ (xanh lá)
        2  # Độ dày chữ
    )

    # Hiển thị frame đã annotate trong cửa sổ
    cv2.imshow("YOLO11 Detection", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()