import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the model
model = YOLO("runs//detect/train28/weights/best.pt")  # Loads trained model


# Load the image
image_path = "./SR_Datasets/V1/test/images/d5ff874e-WIN_20250505_13_07_03_Pro.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib


# Run YOLOv8 inference
results = model(image_path)
# Define colors for each class
colors = {}

# Draw the detected bounding boxes
for result in results:
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]

    print(names)
    for i, (xyxy, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls.int())):
        x1, y1, x2, y2 = map(int, xyxy.tolist())  # Convert to integers

        label = f"{names[i]} {conf:.2f}"  # Class name and confidence

        # Assign a unique color for each class
        if names[i] not in colors:
            colors[names[i]] = tuple(np.random.randint(0, 255, 3).tolist())

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[names[i]], 2)

        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x, text_y = x1, y1 - 5 if y1 - 5 > 10 else y1 + 15
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 3), (text_x + text_size[0], text_y + 3),
                      colors[names[i]], -1)

        # Draw label text
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Display the image with annotations
plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.axis("off")
plt.show()
