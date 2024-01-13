import os
from PIL import Image

def crop_and_save_image(input_path, output_path, target_size=(640, 480)):
    try:
        with Image.open(input_path) as img:
            # Get the original image size
            width, height = img.size

            # Calculate the crop box
            left = (width - target_size[0]) // 2
            top = (height - target_size[1]) // 2
            right = (width + target_size[0]) // 2
            bottom = (height + target_size[1]) // 2

            # Crop and save the image to the output folder
            img = img.crop((left, top, right, bottom))
            img.save(output_path)
    except Exception as e:
        print(f"Error processing image {input_path}: {e}")

def reorganize_crop_and_save_files(output_folder="output"):
    current_dir = os.getcwd() + "\\images"
    output_dir = os.path.join(output_folder)
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for foldername, subfolders, filenames in os.walk(current_dir):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)

            # Create a new filename based on index
            index = filenames.index(filename) + 1
            _, file_extension = os.path.splitext(filename)
            new_filename = f"{index}{file_extension}"

            # Build the new path in the output folder
            new_file_path = os.path.join(output_dir, new_filename)

            # Crop and save the image to the output folder
            crop_and_save_image(file_path, new_file_path)

if __name__ == "__main__":
    reorganize_crop_and_save_files()
    print("Images cropped and saved to the 'output' folder successfully.")
