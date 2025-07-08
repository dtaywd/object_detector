import os
import cv2
import constants as c
from ultralytics import YOLO

def get_model() -> YOLO:
    if (not os.path.exists(c.MODEL_PATH)):
        print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')

    # Load the trained model
    model = YOLO(c.MODEL_PATH)
    return model

def run_model_live_video() -> None:
    model = get_model()

    # Open USB camera (0 = default camera)
    cap = cv2.VideoCapture(0)

    # Set desired camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLO inference on the frame
        results = model(frame)

        # Plot results on the frame
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow("YOLOvX Inference", annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

def run_model_image(image_path: str, save: bool) -> None:
    model = get_model()

    image = cv2.imread(str(image_path))
    results = model(image)
    annotated_image = results[0].plot()

    if save:
        os.makedirs(c.OUTPUT_DIRECTORY_PATH, exist_ok=True)
        out_path = os.path.join(c.OUTPUT_DIRECTORY_PATH, image_path.name)
        cv2.imwrite(out_path, annotated_image)

    cv2.imshow("Image Inference", annotated_image)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    