from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="./SR_Datasets/V1/data.yaml", epochs=100, imgsz=640)


results = model("SR_Datasets/V1/test/images/d5ff874e-WIN_20250505_13_07_03_Pro.jpg")