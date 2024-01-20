# robovision
### State-of-The-Art Computer Vision: Object Detection and Segmentation
![](assets/example.png)

Robovision is a simple, yet powerful software collection that is based on the YOLOv8 model and algorithms.
Out-of-the-box usage is possible using the pretrained weights provided by ultralytics or the open-source community. 
To fine-tune or pretrain a new model, use the labelme tool to label your data, and use datatool to convert the json files into the correct dataset format.

Train the model using:

`python train.py`

Dependencies:

`pip install numpy ultralytics onnxruntime opencv-python torch labelme`

The script calculates the angle of each detected object based on its position relative to the center of the frame. The formula used for angle calculation is as follows:

# Object Angle Calculation Formula

This document provides a simplified explanation of the angle calculation formula used in the script. The formula calculates the angle of each detected object based on its position relative to the center of the frame.

## Object Center Coordinates

The coordinates of the center of the detected object are calculated as follows:

```math
O_x = \frac{\text{box}[0] + \text{box}[2]}{2}
O_y = \frac{\text{box}[1] + \text{box}[3]}{2}
```
## Angle calculation:
```python
delta_x = object_center_x - center_x
delta_y = object_center_y - center_y

angle_rad = math.atan2(delta_y, delta_x)
angle_deg = math.degrees(angle_rad)
```
