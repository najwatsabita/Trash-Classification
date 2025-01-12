import uvicorn
from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms, models
from PIL import Image
import io

app = FastAPI(docs_url="/")

class_mapping = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

class PyTorchPrediction:
    def __init__(self, model_path, class_mapping):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_mapping = class_mapping
        
        self.model = models.resnet18(pretrained=False)
        num_classes = len(class_mapping)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        

        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            
        # Load the state dict into the model
        self.model.load_state_dict(state_dict, strict=True)  
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def prediction(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        pred_prob, pred_idx = torch.max(probabilities, 1)
        pred_idx = pred_idx.item()
        pred_prob = pred_prob.item()
        pred_class = self.class_mapping[pred_idx]
        
        return pred_class, pred_prob, pred_idx


checkpoint_path = 'Trash-Classification.ckpt'  # Replace with your .ckpt file path
leaf_classifier = PyTorchPrediction(checkpoint_path, class_mapping)

@app.post("/prediction/")
async def prediction(file: UploadFile = File(...)):
    contents = await file.read()

    prediction_class_name, prediction_class_prob, prediction_class_idx = (
        leaf_classifier.prediction(contents)
    )
    return {
        "Predicted Class Index": prediction_class_idx,
        "Predicted Class Prob": prediction_class_prob,
        "Predicted Class Names": prediction_class_name
    }

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)