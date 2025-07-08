# consider using deepface https://github.com/serengil/deepface

from ultralytics import YOLO
import constants as c

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11s.pt")

results = model.train(data=c.DATA_YAML_PATH, epochs=40)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("C:\\Users\\David\\Documents\\Git Projects\\Object_detector\\Test photo.jpg")

# List of all detections in the image
boxes = results[0].boxes

# Coordinates of bounding boxes
xyxy = boxes.xyxy  # (x1, y1, x2, y2)

# Confidence scores
conf = boxes.conf

# Class IDs
cls = boxes.cls

# To see results visually
# results[0].show()  # Display annotated image
results[0].save(filename="output.jpg")  # Save annotated image

# # Export the model to ONNX format
# success = model.export(format="onnx")