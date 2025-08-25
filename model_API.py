import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
from typing import Dict, List, Tuple, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

from DensenetModels import DenseNet121

# Global variables to store the loaded model
model = None
device = None

class DetectedDisease(BaseModel):
    disease: str
    probability: float

class PredictionResponse(BaseModel):
    predictions: Dict[str, float]
    detected_diseases: List[DetectedDisease]  # Fixed: Use proper model
    image_name: str
    threshold_used: float

class ErrorResponse(BaseModel):
    error: str
    message: str

def load_model_once(model_path='./models/m-25012018-123527.pth.tar', device_type='cuda'):
    """Load the pre-trained model once at startup - matches ChexnetTrainer.test()"""
    global model, device
    
    if model is not None:
        return model, device
    
    # Model parameters (same as in Main.py and ChexnetTrainer.py)
    nnClassCount = 14
    nnIsTrained = True
    nnArchitecture = 'DENSE-NET-121'
    
    # Determine device
    device = device_type if device_type == 'cuda' and torch.cuda.is_available() else 'cpu'
    
    # Enable cudnn benchmark for consistent performance
    cudnn.benchmark = True
    
    # Create model (same as ChexnetTrainer.test())
    if nnArchitecture == 'DENSE-NET-121': 
        model = DenseNet121(nnClassCount, nnIsTrained)
        if device == 'cuda':
            model = model.cuda()
    
    # Use DataParallel wrapper (same as ChexnetTrainer.test())
    if device == 'cuda':
        model = torch.nn.DataParallel(model).cuda()
    
    # Load checkpoint with key remapping (same as ChexnetTrainer.test())
    modelCheckpoint = torch.load(model_path, map_location=device)
    state_dict = modelCheckpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('.norm.1.', '.norm1.')
        new_k = new_k.replace('.norm.2.', '.norm2.')
        new_k = new_k.replace('.conv.1.', '.conv1.')
        new_k = new_k.replace('.conv.2.', '.conv2.')
        new_state_dict[new_k] = v
    modelCheckpoint['state_dict'] = new_state_dict
    
    # Load the remapped state dict
    model.load_state_dict(modelCheckpoint['state_dict'])
    model.eval()
    
    return model, device

def preprocess_image_from_bytes(image_bytes: bytes, transResize=256, transCrop=224):
    """Preprocess image from bytes using TenCrop transforms - matches ChexnetTrainer.test()"""
    
    # Define transforms exactly as in ChexnetTrainer.test()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    transformList = []
    transformList.append(transforms.Resize(transResize))
    transformList.append(transforms.TenCrop(transCrop))
    transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
    transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
    transformSequence = transforms.Compose(transformList)
    
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transformSequence(image)  # This will have shape [10, 3, 224, 224] for TenCrop
    
    return image_tensor

def predict_disease_from_tensor(image_tensor, threshold=0.5):
    """
    Predict diseases from preprocessed image tensor using the same approach as ChexnetTrainer.test()
    """
    global model, device
    
    # Disease class names (same as in ChexnetTrainer.test())
    CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                   'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    
    if device == 'cuda':
        image_tensor = image_tensor.cuda()
    
    # Make prediction (same logic as ChexnetTrainer.test())
    with torch.no_grad():
        # Handle TenCrop: image_tensor has shape [10, 3, 224, 224]
        n_crops, c, h, w = image_tensor.size()
        
        # Reshape for model input
        input_tensor = image_tensor.view(-1, c, h, w)
        if device == 'cuda':
            input_tensor = input_tensor.cuda()
        
        # Get model output
        out = model(input_tensor)
        
        # Average over the 10 crops (same as ChexnetTrainer.test())
        outMean = out.view(1, n_crops, -1).mean(1)  # [1, 14]
        
        probabilities = outMean.cpu().numpy()[0]  # Remove batch dimension
    
    # Create results
    predictions = {}
    detected_diseases = []
    
    for i, class_name in enumerate(CLASS_NAMES):
        prob = float(probabilities[i])
        predictions[class_name] = prob
        
        if prob >= threshold:
            # Fixed: Use proper DetectedDisease model
            detected_diseases.append(DetectedDisease(disease=class_name, probability=prob))
    
    return predictions, detected_diseases

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    try:
        print("Loading model at startup...")
        load_model_once()
        print(f"Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        raise e
    
    yield
    
    # Shutdown (cleanup if needed)
    print("Shutting down API...")

# FastAPI app instance with lifespan handler
app = FastAPI(
    title="ChestX-ray Disease Classification API",
    description="API for classifying chest X-ray images using DenseNet121",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "ChestX-ray Disease Classification API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global model, device
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_disease_endpoint(
    file: UploadFile = File(...),
    threshold: Optional[float] = 0.5
):
    """
    Predict diseases from uploaded chest X-ray image
    
    Args:
        file: Uploaded image file (JPG, PNG, etc.)
        threshold: Threshold for binary classification (default: 0.5)
    
    Returns:
        PredictionResponse with predictions and detected diseases
    """
    global model, device
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate threshold
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
    
    try:
        # Ensure model is loaded
        if model is None:
            load_model_once()
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        image_tensor = preprocess_image_from_bytes(image_bytes)
        
        # Make prediction
        predictions, detected_diseases = predict_disease_from_tensor(image_tensor, threshold)
        
        return PredictionResponse(
            predictions=predictions,
            detected_diseases=detected_diseases,
            image_name=file.filename,
            threshold_used=threshold
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch_endpoint(
    files: List[UploadFile] = File(...),
    threshold: Optional[float] = 0.5
):
    """
    Predict diseases from multiple uploaded chest X-ray images
    
    Args:
        files: List of uploaded image files
        threshold: Threshold for binary classification (default: 0.5)
    
    Returns:
        List of predictions for each image
    """
    global model, device
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    # Validate threshold
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
    
    results = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "error": "File must be an image"
            })
            continue
        
        try:
            # Ensure model is loaded
            if model is None:
                load_model_once()
            
            # Read image bytes
            image_bytes = await file.read()
            
            # Preprocess image
            image_tensor = preprocess_image_from_bytes(image_bytes)
            
            # Make prediction
            predictions, detected_diseases = predict_disease_from_tensor(image_tensor, threshold)
            
            # Convert DetectedDisease objects to dictionaries for JSON response
            detected_diseases_dict = [{"disease": d.disease, "probability": d.probability} for d in detected_diseases]
            
            results.append({
                "filename": file.filename,
                "predictions": predictions,
                "detected_diseases": detected_diseases_dict,
                "threshold_used": threshold
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Prediction failed: {str(e)}"
            })
    
    return {"results": results}

def main():
    """Run the FastAPI server"""
    print("Starting ChestX-ray Disease Classification API...")
    uvicorn.run(
        "model_API:app",  # module:app
        host="0.0.0.0",   # Allow external connections
        port=8000,        # Default port
        reload=True       # Auto-reload on code changes (for development)
    )

if __name__ == "__main__":
    main()