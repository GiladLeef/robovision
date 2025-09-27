import os
from PIL import Image

def cropAndSaveImage(inputPath, outputPath, targetSize=(640, 480)):
    try:
        with Image.open(inputPath) as img:
            width, height = img.size

            left = (width - targetSize[0]) // 2
            top = (height - targetSize[1]) // 2
            right = (width + targetSize[0]) // 2
            bottom = (height + targetSize[1]) // 2

            img = img.crop((left, top, right, bottom))
            img.save(outputPath)
    except Exception as e:
        print(f"Error processing image {inputPath}: {e}")

def reorganizeCropAndSaveFiles(outputFolder="output"):
    currentDir = os.getcwd() + "\\images"
    outputDir = os.path.join(outputFolder)
    
    os.makedirs(outputDir, exist_ok=True)

    for foldername, subfolders, filenames in os.walk(currentDir):
        for filename in filenames:
            filePath = os.path.join(foldername, filename)

            index = filenames.index(filename) + 1
            _, fileExtension = os.path.splitext(filename)
            newFilename = f"{index}{fileExtension}"

            newFilePath = os.path.join(outputDir, newFilename)

            cropAndSaveImage(filePath, newFilePath)

if __name__ == "__main__":
    reorganizeCropAndSaveFiles()
    print("Images cropped and saved to the 'output' folder successfully.")
