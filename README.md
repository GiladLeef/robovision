# robovision
### State-of-The-Art Computer Vision: Object Detection and Segmentation
![](assets/example.png)

Robovision is a simple, yet powerful software collection that is based on the YOLOv8 model and algorithms.
Out-of-the-box usage is possible using the pretrained weights provided by ultralytics or the open-source community. 
To fine-tune or pretrain a new model, use the labelme tool to label your data, and use datatool to convert the json files into the correct dataset format.

`python train.py`

Use labelme to easily label your data in a click of a button. 

Dependencies:

`pip install numpy ultralytics onnxruntime opencv-python torch labelme`
