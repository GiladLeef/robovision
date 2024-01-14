import cv2
import numpy as np
import os
import random

def mask_color_object(image):
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color in HSV
    lower_red_orange = np.array([0, 150, 100])
    upper_red_orange = np.array([20, 255, 255])
    # Create a mask for the color
    mask = cv2.inRange(hsv, lower_red_orange, upper_red_orange)

    # Bitwise-AND operation to extract the object
    color_object = cv2.bitwise_and(image, image, mask=mask)

    return color_object, mask

def filter_contours(contours, min_area=6000):
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            filtered_contours.append(contour)
    return filtered_contours

def save_polygon_to_txt(polygon, filepath, image_width, image_height):
    # Save the polygon vertices to a text file in YOLO format
    with open(filepath, 'w') as file:
        file.write("0 ")  # Class index (0 for your case)
        for point in polygon:
            x, y = point

            # Normalize coordinates to be between 0 and 1
            normalized_x = x / image_width
            normalized_y = y / image_height

            file.write(f"{normalized_x} {normalized_y} ")

        file.write('\n')

def mask_image_based_on_polygon(image, polygon):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [polygon], (255, 255, 255)) 
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def split_dataset(images_folder, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # Get the list of image files
    image_files = [file for file in os.listdir(images_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]

    # Shuffle the list for random splitting
    random.shuffle(image_files)

    # Calculate the number of images for each split
    total_images = len(image_files)
    num_train = int(total_images * train_ratio)
    num_val = int(total_images * val_ratio)
    num_test = total_images - num_train - num_val

    # Split the image files
    train_files = image_files[:num_train]
    val_files = image_files[num_train:num_train+num_val]
    test_files = image_files[num_train+num_val:]

    return train_files, val_files, test_files

def create_dataset_folders(base_folder):
    # Create the dataset structure
    dataset_folder = os.path.join(base_folder, 'images')
    train_folder = os.path.join(dataset_folder, 'train')
    val_folder = os.path.join(dataset_folder, 'val')
    test_folder = os.path.join(dataset_folder, 'test')

    # Create train, val, and test folders if they don't exist
    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(folder, exist_ok=True)

    return train_folder, val_folder, test_folder

def process_images(input_folder, output_folder, labels_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over each file in the input folder and its subfolders
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Read the image
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)

                # Apply the color object masking
                _, mask = mask_color_object(image)

                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filter contours based on area (consider as shapes)
                filtered_contours = filter_contours(contours)
                # Loop through filtered contours
                for idx, contour in enumerate(filtered_contours):
                    # Convert the contour to a polygon
                    polygon = contour.squeeze()

                    # Save the polygon to a .txt file inside the 'labels' folder
                    filename_without_extension, _ = os.path.splitext(filename)
                    txt_filename = f"{filename_without_extension}.txt"
                    txt_filepath = os.path.join(labels_folder, txt_filename)

                    # Get image width and height for normalization
                    image_width, image_height = image.shape[1], image.shape[0]

                    # Save the polygon in YOLO format
                    save_polygon_to_txt(polygon, txt_filepath, image_width, image_height)

                    # Mask the image based on the polygon and save the result
                    masked_image = mask_image_based_on_polygon(image, polygon)
                    output_image_path = os.path.join(output_folder, filename)
                    output__mask_image_path = os.path.join(output_folder, "mask_" + filename)
                    cv2.imwrite(output_image_path, image)
                    cv2.imwrite(output__mask_image_path, masked_image)

if __name__ == "__main__":
    # Specify the input and output folders
    input_folder = "images"
    output_folder = "output"

    # Create the dataset folders
    train_folder, val_folder, test_folder = create_dataset_folders(output_folder)

    # Create the labels folders directly under 'output/labels'
    train_labels_folder = os.path.join(output_folder, 'labels', 'train')
    val_labels_folder = os.path.join(output_folder, 'labels', 'val')
    test_labels_folder = os.path.join(output_folder, 'labels', 'test')

    for folder in [train_labels_folder, val_labels_folder, test_labels_folder]:
        os.makedirs(folder, exist_ok=True)

    # Split the dataset into train, val, and test sets
    train_files, val_files, test_files = split_dataset(input_folder)

    # Process the images in the input folder and its subfolders
    process_images(input_folder, train_folder, train_labels_folder)
    process_images(input_folder, val_folder, val_labels_folder)
    process_images(input_folder, test_folder, test_labels_folder)
