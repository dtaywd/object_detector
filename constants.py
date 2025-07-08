import os

# paths
CWD = os.getcwd()
YOLO_LABELS_FOLDER_PATH = os.path.join(CWD, "yolo_labels")
DATA_FOLDER_PATH = os.path.join(CWD, "data")

INPUT_IMAGE_FOLDER_PATH = os.path.join(YOLO_LABELS_FOLDER_PATH,'images')
INPUT_LABEL_FOLDER_PATH = os.path.join(YOLO_LABELS_FOLDER_PATH,'labels')

TRAIN_IMG_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "train/images")
TRAIN_LABEL_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "train/labels")
VAL_IMG_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "validation/images")
VAL_LABEL_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "validation/labels")

LABELS_PATH = os.path.join(YOLO_LABELS_FOLDER_PATH, "classes.txt")
DATA_YAML_PATH = os.path.join(CWD, "data.yaml")

OUTPUT_DIRECTORY_PATH = os.path.join(CWD, "output")

# constants
TRAINING_PROPORTION = 0.9

# using trained model
MODEL_PATH = os.path.join(CWD, "runs/detect/train4/weights/best.pt")
