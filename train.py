from ultralytics import YOLO
def main():
    # Load pretrained model weights
    # model = YOLO('robovision\\models\\best.pt')
    
    # Pretrain from scratch. load configuration from the model.yaml file:
    model = YOLO('dataset\\model.yaml')  

    # Train the model
    results = model.train(data='dataset\\dataset.yaml', epochs=100, imgsz=640)

if __name__ == "__main__":
    main()
