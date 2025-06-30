import os
import random
import shutil

# consider using deepface https://github.com/serengil/deepface
def split_data_into_training_and_validation(training_proportion: int) -> None:
    # define paths
    cwd = os.getcwd()
    yolo_labels_folder_path = os.path.join(cwd, "yolo_labels")
    data_folder_path = os.path.join(cwd, "data")

    input_image_folder_path = os.path.join(yolo_labels_folder_path,'images')
    input_label_folder_path = os.path.join(yolo_labels_folder_path,'labels')

    train_img_folder_path = os.path.join(data_folder_path, "train/images")
    train_label_folder_path = os.path.join(data_folder_path, "train/labels")
    val_img_folder_path = os.path.join(data_folder_path, "validation/images")
    val_label_folder_path = os.path.join(data_folder_path, "validation/labels")

    # Create output directories
    for dir_path in [train_img_folder_path, train_label_folder_path, val_img_folder_path, val_label_folder_path]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f'Created folder at {dir_path}.')

    image_files = [f for f in os.listdir(input_image_folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_files.sort()
    random.seed(42)
    random.shuffle(image_files)
    num_total = len(image_files)
    split_idx = int(num_total * training_proportion)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    copy_files(
        file_list=train_files,
        input_image_folder_path=input_image_folder_path,
        dest_images_folder_path=train_img_folder_path,
        input_label_folder_path=input_label_folder_path,
        dest_labels_folder_path=train_label_folder_path
    )

    copy_files(
        file_list=val_files,
        input_image_folder_path=input_image_folder_path,
        dest_images_folder_path=val_img_folder_path,
        input_label_folder_path=input_label_folder_path,
        dest_labels_folder_path=val_label_folder_path
    )


def copy_files(file_list, input_image_folder_path, dest_images_folder_path, input_label_folder_path, dest_labels_folder_path):
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

split_data_into_training_and_validation(training_proportion=0.9)