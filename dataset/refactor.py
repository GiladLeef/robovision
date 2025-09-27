import os
import shutil

datasetFolder = 'images'

outputFolder = 'output'

os.makedirs(outputFolder, exist_ok=True)

allFiles = os.listdir(datasetFolder)

filePairs = {}

for file in allFiles:
    name, extension = os.path.splitext(file)
    if extension == '.jpg' or extension == '.json':
        filePairs.setdefault(name, {})[extension] = file

sortedKeys = sorted(filePairs.keys(), key=lambda x: int(x))

for i, key in enumerate(sortedKeys, start=1):
    pair = filePairs[key]
    
    jpgFile = pair.get('.jpg')
    jsonFile = pair.get('.json')

    if jpgFile and jsonFile:
        newJpgName = f'{i}.jpg'
        newJsonName = f'{i}.json'

        jpgPath = os.path.join(datasetFolder, jpgFile)
        jsonPath = os.path.join(datasetFolder, jsonFile)

        newJpgPath = os.path.join(outputFolder, newJpgName)
        newJsonPath = os.path.join(outputFolder, newJsonName)

        shutil.copy(jpgPath, newJpgPath)
        shutil.copy(jsonPath, newJsonPath)

        print(f'Copied: {jpgFile} to {newJpgName}, {jsonFile} to {newJsonName}')
