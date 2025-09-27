from ultralytics import YOLO

def main():

    # Load pretrained model weights
    # model = YOLO('robovision\\models\\best.pt')

    # Train the model
    model = YOLO('dataset\\model.yaml')  
    results = model.train(data='dataset\\dataset.yaml', epochs=100, imgsz=640)

if __name__ == "__main__":
    main()
