import os
import random
import shutil
import yaml
import constants as c

def split_data_into_training_and_validation(training_proportion: int) -> None:
    create_folders()
    
    image_files = [f for f in os.listdir(c.INPUT_IMAGE_FOLDER_PATH) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_files.sort()
    random.seed(42)
    random.shuffle(image_files)
    num_total = len(image_files)
    split_idx = int(num_total * training_proportion)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    copy_files(
        file_list=train_files,
        input_image_folder_path=c.INPUT_IMAGE_FOLDER_PATH,
        dest_images_folder_path=c.TRAIN_IMG_FOLDER_PATH,
        input_label_folder_path=c.INPUT_LABEL_FOLDER_PATH,
        dest_labels_folder_path=c.TRAIN_LABEL_FOLDER_PATH
    )

    copy_files(
        file_list=val_files,
        input_image_folder_path=c.INPUT_IMAGE_FOLDER_PATH,
        dest_images_folder_path=c.VAL_IMG_FOLDER_PATH,
        input_label_folder_path=c.INPUT_LABEL_FOLDER_PATH,
        dest_labels_folder_path=c.VAL_LABEL_FOLDER_PATH
    )

def create_folders() -> None:
    # should update this to include all folders to be created
    for dir_path in [c.TRAIN_IMG_FOLDER_PATH, c.TRAIN_LABEL_FOLDER_PATH, c.VAL_IMG_FOLDER_PATH, c.VAL_LABEL_FOLDER_PATH]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f'Created folder at {dir_path}.')

def copy_files(file_list: str, input_image_folder_path: str, dest_images_folder_path: str, input_label_folder_path: str, dest_labels_folder_path: str) -> None:
    for img_filename in file_list:
        # Copy image
        src_img = os.path.join(input_image_folder_path, img_filename)
        dst_img = os.path.join(dest_images_folder_path, img_filename)
        shutil.copy2(src_img, dst_img)

        # Copy corresponding label
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        src_label = os.path.join(input_label_folder_path, label_filename)
        dst_label = os.path.join(dest_labels_folder_path, label_filename)

        # Some images may not have labels (depending on dataset)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

def create_data_yaml(labels_path: str, data_yaml_path: str) -> None:
  # Read class.txt to get class names
    if not os.path.exists(labels_path):
        print(f'classes.txt file not found! Please create a classes.txt labelmap and move it to {labels_path}')
        return
    with open(labels_path, 'r') as f:
        classes = []
        for line in f.readlines():
            if len(line.strip()) == 0: 
                continue
            classes.append(line.strip())
    
    number_of_classes = len(classes)

    # Create data dictionary
    data = {
        'path': '/data',
        'train': 'train/images',
        'val': 'validation/images',
        'nc': number_of_classes,
        'names': classes
    }

    # Write data to YAML file
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
        print(f'Created config file at {data_yaml_path}')

if __name__ == "__main__":
    split_data_into_training_and_validation(training_proportion=c.TRAINING_PROPORTION)
    create_data_yaml(labels_path=c.LABELS_PATH, data_yaml_path=c.DATA_YAML_PATH)