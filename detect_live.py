import time

import cv2
import numpy as np
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("runs//detect/train28/weights/best.pt")  # Ensure your trained model exists

import cv2

idx = 0

"""for i in range(10):  # Check indices 0-9
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera available at index {i}")
        idx = i
        cap.release()
        break"""
# Open the webcam (0 = default camera, change if using an external camera)
cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)


# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Define colors for each class
colors = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Get image dimensions
    height, width, _ = frame.shape

    # Compute new starting x position (crop 20% from left)
    new_x_start = int(width * 0.2)  # 20% of width
    cropped_frame = frame[:, new_x_start:]  # Crop left side



    # Run YOLO inference
    cv2.imwrite("test.jpg", cropped_frame)  # Save a test frame
    results = model("test.jpg")  # Run YOLO on the saved frame

    # Process results and draw detections
    for result in results:
        for i, (xyxy, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls.int())):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            label = f"{result.names[cls.item()]} {conf:.2f}"
            x1 += new_x_start
            x2 += new_x_start
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the live video
    cv2.imshow("YOLO Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'Q' to exit

    time.sleep(1)

# Release resources
cap.release()
cv2.destroyAllWindows()
