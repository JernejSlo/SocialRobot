from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="./datasets/data.yaml", epochs=1, imgsz=640)


results = model("datasets/test/images/image-13-_jpeg.rf.c2b0cd074dcef71305a8864fa65fb9c2.jpg")