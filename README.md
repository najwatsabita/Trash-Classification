## Trash Classification 

### Overview
This project implements a trash classification system capable of identifying 6 different types of waste materials (cardboard, glass, metal, paper, plastic, and trash). The system uses computer vision to detect and classify waste items in images, providing bounding box annotations and confidence scores for each detection.

### Model Details 
1. Trained model checkpoint (.ckpt format)
2. Training completed over 20 epochs (You can adjust by yourself)
3. Implements object detection with classification capabilities

### Usage
1. Prepare your input image containing waste items
2. Run the program with your image
3. The system will process the image and output:
    a. Bounding boxes around detected waste items
    b. Labels showing the waste classification
    c. Confidence scores for each detection
    d. You can save your output image to your local storage

### Requirements Installation
```bash
pip install -r requirements.txt
```

### Dataset
#### Resize Dataset
The dataset was resized to 224x224 from its original size.
#### Data Split
The dataset was divided into 80% for training 20% for validation.
#### Dataset Issue
The dataset was initially compatible with MacOS, but since I used Windows, it did not work at first so I need to clean it first.

### Model 
#### Architecture 
1. Framework : Pytorch Lightning
2. Base Model : ResNet18 (pretrained)
3. Optimizer : Adam (Learning rate : 0.001)


### Training with WandB Monitoring 
1. First, You need to create WandB account in WandB Official Website. 
2. And then when you run the training process you will get the link to get the special key from WandB 
3. then paste in it's place. WandB can monitoring your training step by giving a chart like epochs, train_val, train_loss, val_acc, val_loss, etc. 
### Usage 
```bash
model = ImageClassifier(num_classes=num_classes) #Replace with the appropriate number of classes
```

```bash
trainer = pl.Trainer(
    max_epochs=20,
    callbacks=[checkpoint_callback, early_stopping],
    logger=wandb_logger
)

trainer.fit(
    model, 
    train_loader, 
    val_loader,
)
```
### Evaluate Model 
You can evaluate your model by checking the accuracy, accuracy will appear in percentage in decimal form
#### Accuracy Calculation
```bash
print(f"Validation Accuracy: {acc/len(val_loader):.4f}")
```

### Testing 
1. Prepare the images for testing 
2. Replace the path of images
3. Run the code
4. Your output will save in predicted_images folder

### Output 


