import os
import shutil

# Specify the path to your dataset folder
dataset_folder = 'images'

# Specify the path to the output folder
output_folder = 'output'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of all files in the folder
all_files = os.listdir(dataset_folder)

# Create a dictionary to store the pairs
file_pairs = {}

# Iterate through all files and group them by the common number in the filename
for file in all_files:
    name, extension = os.path.splitext(file)
    if extension == '.jpg' or extension == '.json':
        file_pairs.setdefault(name, {})[extension] = file

# Sort the keys (common numbers) in the dictionary
sorted_keys = sorted(file_pairs.keys(), key=lambda x: int(x))

# Iterate through the sorted keys and copy the files to the output folder
for i, key in enumerate(sorted_keys, start=1):
    pair = file_pairs[key]
    
    jpg_file = pair.get('.jpg')
    json_file = pair.get('.json')

    if jpg_file and json_file:
        new_jpg_name = f'{i}.jpg'
        new_json_name = f'{i}.json'

        jpg_path = os.path.join(dataset_folder, jpg_file)
        json_path = os.path.join(dataset_folder, json_file)

        new_jpg_path = os.path.join(output_folder, new_jpg_name)
        new_json_path = os.path.join(output_folder, new_json_name)

        shutil.copy(jpg_path, new_jpg_path)
        shutil.copy(json_path, new_json_path)

        print(f'Copied: {jpg_file} to {new_jpg_name}, {json_file} to {new_json_name}')
