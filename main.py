from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import io
import json
import os
from typing import Dict, Any
import uvicorn
from contextlib import asynccontextmanager

# Your existing model class
class LargeScaleProductClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b4', num_classes=7, num_groups=6, pretrained=True):
        super(LargeScaleProductClassifier, self).__init__()
        
        # Use timm for access to latest models
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get the number of features from the backbone
        num_features = self.backbone.num_features
        
        # Shared feature processing
        self.shared_layers = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.3),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )
        
        # Task-specific heads
        self.class_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        self.group_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, num_groups)
        )
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Process through shared layers
        shared_features = self.shared_layers(features)
        
        # Get predictions from both heads
        class_output = self.class_head(shared_features)
        group_output = self.group_head(shared_features)
        
        return class_output, group_output

# Simplified predictor class for API
class ProductRecognitionAPI:
    def __init__(self, model_path: str, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        self.model_name = self.checkpoint.get('model_name', 'efficientnet_b4')
        self.num_classes = self.checkpoint['num_classes']
        self.num_groups = self.checkpoint['num_groups']
        
        # Initialize model
        self.model = LargeScaleProductClassifier(
            model_name=self.model_name,
            num_classes=self.num_classes,
            num_groups=self.num_groups,
            pretrained=False
        ).to(self.device)
        
        # Load the trained weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize default mappings
        self.class_mapping = {i: f"Class_{i}" for i in range(self.num_classes)}
        self.group_mapping = {i: f"Group_{i}" for i in range(self.num_groups)}
        
        # Try to load mappings if they exist
        self.load_mappings()
    
    def load_mappings(self):
        """Load class and group mappings if they exist"""
        if os.path.exists('class_mapping.json'):
            with open('class_mapping.json', 'r') as f:
                self.class_mapping = json.load(f)
                # Convert string keys to int
                self.class_mapping = {int(k): v for k, v in self.class_mapping.items()}
        
        if os.path.exists('group_mapping.json'):
            with open('group_mapping.json', 'r') as f:
                self.group_mapping = json.load(f)
                # Convert string keys to int
                self.group_mapping = {int(k): v for k, v in self.group_mapping.items()}
    
    def predict_image(self, image: Image.Image) -> Dict[str, Any]:
        """Predict class and group for a single image"""
        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            class_outputs, group_outputs = self.model(input_tensor)
            
            # Get probabilities
            class_probs = torch.softmax(class_outputs, dim=1)
            group_probs = torch.softmax(group_outputs, dim=1)
            
            # Get predicted classes
            class_pred = torch.argmax(class_outputs, dim=1).item()
            group_pred = torch.argmax(group_outputs, dim=1).item()
            
            # Get confidence scores
            class_confidence = class_probs[0][class_pred].item()
            group_confidence = group_probs[0][group_pred].item()
        
        # Convert predictions to labels
        predicted_class = self.class_mapping.get(class_pred)
        predicted_group = self.group_mapping.get(group_pred)
        
        return {
            'predicted_class': predicted_class,
            'predicted_group': predicted_group,
            'class_confidence': round(class_confidence, 4),
            'group_confidence': round(group_confidence, 4),
            'class_index': class_pred,
            'group_index': group_pred
        }

# Global predictor instance
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model when the server starts
    global predictor
    model_path = os.getenv('MODEL_PATH', 'best_product_model_efficientnet_b4.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    predictor = ProductRecognitionAPI(model_path)
    print(f"Model loaded successfully from {model_path}")
    yield
    # Clean up (if needed)
    predictor = None

# Create FastAPI app
app = FastAPI(
    title="Product Classification API",
    description="API for classifying products into classes and groups",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Product Classification API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict")
async def predict_product(file: UploadFile = File(...)):
    """
    Predict product class and group from an uploaded image
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Make prediction
        result = predictor.predict_image(image)
        
        return JSONResponse(content={
            "success": True,
            "prediction": result,
            "filename": file.filename
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_name": predictor.model_name,
        "num_classes": predictor.num_classes,
        "num_groups": predictor.num_groups,
        "device": str(predictor.device),
        "classes": predictor.class_mapping,
        "groups": predictor.group_mapping
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )