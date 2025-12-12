import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from misc_functions import preprocess_image
from pytorchcv.model_provider import get_model as ptcv_get_model
from cam_utils import ClassSpecificImageGeneration

def analyze_class_visualization(model, image_path, top_k=5):
    """
    Analyze the classification probabilities for a class visualization image.
    
    Args:
        model: The neural network model
        image_path: Path to the class visualization image
        top_k: Number of top probabilities to return (default: 5)
    
    Returns:
        list: List of tuples containing (class_idx, probability) for top k classes
    """
    # Load and preprocess the image
    try:
        img = Image.open(image_path).convert('RGB')  # Ensure RGB mode
        processed_img = preprocess_image(img, False)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return []
    
    # Move to same device as model
    device = next(model.parameters()).device
    processed_img = processed_img.to(device)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        output = model(processed_img)
        # Handle tuple output (logits, features)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
            
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get top k probabilities and indices
        top_probs, top_indices = torch.topk(probs[0], top_k)
        
        # Convert to list of tuples (class_idx, probability)
        results = [(idx.item(), prob.item()) for idx, prob in zip(top_indices, top_probs)]
        
    return results

def analyze_visualization_directory(model, directory_path, top_k=5):
    """
    Analyze all class visualization images in a directory.
    
    Args:
        model: The neural network model
        directory_path: Path to directory containing class visualization images
        top_k: Number of top probabilities to return (default: 5)
    
    Returns:
        dict: Dictionary mapping image filenames to their top k predictions
    """
    results = {}
    
    # Get all PNG files in directory
    image_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]
    
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        predictions = analyze_class_visualization(model, image_path, top_k)
        results[image_file] = predictions
        
        # Print results for this image
        print(f"\nResults for {image_file}:")
        for class_idx, prob in predictions:
            print(f"Class {class_idx}: {prob:.4f}")
            
    return results

if __name__ == "__main__":
    # Example usage
    from models import *
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze class visualization probabilities')
    parser.add_argument('--model', type=str, default='resnet18', help='Model name')
    parser.add_argument('--directory', type=str, default='./new2/pruned/class_visualizations', 
                      help='Path to directory containing class visualization PNG files')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top probabilities to show')
    args = parser.parse_args()
    
    # Load model
    model = ptcv_get_model(args.model, pretrained=True)
    model.cuda()
    model.eval()

    viz_save_dir = os.path.join("./new2/origin/resnet18", 'class_visualizations')
    if not os.path.exists(viz_save_dir):
        os.makedirs(viz_save_dir)
            
        # Generate visualizations for both original and pruned models
    print("Generating visualizations for original model...")
    for i in range(710,720):
        csig_origin = ClassSpecificImageGeneration(model, i, save_dir=viz_save_dir)
        csig_origin.generate()
    
    # Analyze visualizations
    results = analyze_visualization_directory(model, args.directory, args.top_k) 