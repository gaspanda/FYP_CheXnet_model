from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import your model components
from DensenetModels import DenseNet121

app = Flask(__name__)
CORS(app)  # Enable CORS for external API calls

# Global variables
model = None
device = None

# Disease class names
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

def load_model_once():
    """Load the pre-trained model once at startup"""
    global model, device
    
    if model is not None:
        return model, device
    
    # Model parameters
    nnClassCount = 14
    nnIsTrained = True
    nnArchitecture = 'DENSE-NET-121'
    
    # Use CPU for Render deployment (GPU not available on free tier)
    device = 'cpu'
    
    # Create model
    if nnArchitecture == 'DENSE-NET-121': 
        model = DenseNet121(nnClassCount, nnIsTrained)
    
    # Load checkpoint with key remapping
    model_path = './models/m-25012018-123527.pth.tar'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    modelCheckpoint = torch.load(model_path, map_location=device)
    state_dict = modelCheckpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('.norm.1.', '.norm1.')
        new_k = new_k.replace('.norm.2.', '.norm2.')
        new_k = new_k.replace('.conv.1.', '.conv1.')
        new_k = new_k.replace('.conv.2.', '.conv2.')
        new_state_dict[new_k] = v
    
    # Remove 'module.' prefix if present (from DataParallel)
    final_state_dict = {}
    for k, v in new_state_dict.items():
        if k.startswith('module.'):
            final_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            final_state_dict[k] = v
    
    model.load_state_dict(final_state_dict)
    model.eval()
    
    return model, device

def preprocess_image_from_bytes(image_bytes, transResize=256, transCrop=224):
    """Preprocess image from bytes using TenCrop transforms"""
    
    # Define transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    transformList = []
    transformList.append(transforms.Resize(transResize))
    transformList.append(transforms.TenCrop(transCrop))
    transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
    transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
    transformSequence = transforms.Compose(transformList)
    
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transformSequence(image)  # Shape: [10, 3, 224, 224] for TenCrop
    
    return image_tensor

def predict_disease_from_tensor(image_tensor, threshold=0.5):
    """Predict diseases from preprocessed image tensor"""
    global model, device
    
    with torch.no_grad():
        # Move tensor to device
        if device == 'cuda':
            image_tensor = image_tensor.cuda()
        
        # Get predictions for all 10 crops
        outputs = model(image_tensor)
        
        # Average predictions across all crops
        outputs_mean = outputs.view(-1, len(CLASS_NAMES)).mean(0)
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(outputs_mean)
        
        # Convert to numpy
        probs_np = probabilities.cpu().numpy()
    
    # Create prediction dictionary
    predictions = {}
    detected_diseases = []
    
    for i, class_name in enumerate(CLASS_NAMES):
        prob = float(probs_np[i])
        predictions[class_name] = prob
        
        if prob >= threshold:
            detected_diseases.append({
                'disease': class_name,
                'probability': prob
            })
    
    return predictions, detected_diseases

@app.route('/', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "ChestX-ray Disease Classification API is running",
        "model_loaded": model is not None
    })

@app.route('/health', methods=['GET'])
def detailed_health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "service": "ChestX-ray Disease Classification API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "device": device,
        "classes": CLASS_NAMES
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict diseases from uploaded chest X-ray image"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get threshold parameter
        threshold = float(request.form.get('threshold', 0.5))
        
        # Read image bytes
        image_bytes = file.read()
        
        # Preprocess image
        image_tensor = preprocess_image_from_bytes(image_bytes)
        
        # Make prediction
        predictions, detected_diseases = predict_disease_from_tensor(image_tensor, threshold)
        
        # Return response
        return jsonify({
            'predictions': predictions,
            'detected_diseases': detected_diseases,
            'image_name': file.filename,
            'threshold_used': threshold
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict diseases for multiple images"""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        threshold = float(request.form.get('threshold', 0.5))
        results = []
        
        for file in files:
            if file.filename != '':
                image_bytes = file.read()
                image_tensor = preprocess_image_from_bytes(image_bytes)
                predictions, detected_diseases = predict_disease_from_tensor(image_tensor, threshold)
                
                results.append({
                    'image_name': file.filename,
                    'predictions': predictions,
                    'detected_diseases': detected_diseases,
                    'threshold_used': threshold
                })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

# Initialize model on startup
with app.app_context():
    try:
        load_model_once()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")

# if __name__ == '__main__':
#     # For local testing
#     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)