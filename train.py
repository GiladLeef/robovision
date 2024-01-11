from ultralytics import YOLO
def main():
    # Load a model
    model = YOLO('dataset\\model.yaml')  # build a new model from YAML
    # Train the model
    results = model.train(data='dataset\\dataset.yaml', epochs=100, imgsz=640)

if __name__ == "__main__":
    main()
