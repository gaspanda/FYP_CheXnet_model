import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from DensenetModels import DenseNet121

def load_model(model_path, device='cuda'):
    """Load the pre-trained model with key mapping for compatibility - matches ChexnetTrainer.test()"""
    
    # Model parameters (same as in Main.py and ChexnetTrainer.py)
    nnClassCount = 14
    nnIsTrained = True
    nnArchitecture = 'DENSE-NET-121'
    
    # Enable cudnn benchmark for consistent performance
    cudnn.benchmark = True
    
    # Create model (same as ChexnetTrainer.test())
    if nnArchitecture == 'DENSE-NET-121': 
        model = DenseNet121(nnClassCount, nnIsTrained)
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
    
    # Use DataParallel wrapper (same as ChexnetTrainer.test())
    if device == 'cuda' and torch.cuda.is_available():
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
    
    return model

def preprocess_image(image_path, transResize=256, transCrop=224):
    """Preprocess the input image using TenCrop transforms - matches ChexnetTrainer.test()"""
    
    # Define transforms exactly as in ChexnetTrainer.test()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    transformList = []
    transformList.append(transforms.Resize(transResize))
    transformList.append(transforms.TenCrop(transCrop))
    transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
    transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
    transformSequence = transforms.Compose(transformList)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transformSequence(image)  # This will have shape [10, 3, 224, 224] for TenCrop
    
    return image_tensor

def predict_disease(image_path, model_path='./models/m-25012018-123527.pth.tar', threshold=0.5):
    """
    Predict diseases from a chest X-ray image using the same approach as ChexnetTrainer.test()
    
    Args:
        image_path: Path to the input image
        model_path: Path to the trained model
        threshold: Threshold for binary classification (default: 0.5)
    
    Returns:
        predictions: Dictionary with disease names and their probabilities
        detected_diseases: List of diseases detected above threshold
    """
    
    # Disease class names (same as in ChexnetTrainer.test())
    CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                   'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    
    # Parameters from Main.py
    transResize = 256
    transCrop = 224
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Preprocess image
    print("Preprocessing image...")
    image_tensor = preprocess_image(image_path, transResize, transCrop)
    
    if device == 'cuda':
        image_tensor = image_tensor.cuda()
    
    # Make prediction (same logic as ChexnetTrainer.test())
    print("Making prediction...")
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
            detected_diseases.append((class_name, prob))
    
    return predictions, detected_diseases

def main():
    """Example usage with command line arguments"""
    import sys
    
    # Default image path
    default_image_path = "./test/00009285_000.png"
    
    # Get image path from command line argument or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = default_image_path
        print(f"No image path provided, using default: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Usage: python run_model.py [image_path]")
        print(f"Example: python run_model.py ./test/00009285_000.png")
        return
    
    try:
        # Make prediction
        predictions, detected_diseases = predict_disease(image_path)
        
        print(f"\nResults for image: {image_path}")
        print("=" * 50)
        
        # Print all probabilities
        print("\nAll disease probabilities:")
        for disease, prob in predictions.items():
            print(f"{disease:20}: {prob:.4f}")
        
        # Print detected diseases (above threshold)
        print(f"\nDetected diseases (probability >= 0.5):")
        if detected_diseases:
            for disease, prob in detected_diseases:
                print(f"- {disease}: {prob:.4f}")
        else:
            print("No diseases detected above threshold")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
