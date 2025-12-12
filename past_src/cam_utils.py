import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime

import torch
from torch.optim import SGD
from torchvision import models

from misc_functions import preprocess_image, recreate_image, save_image


class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class, save_dir='./generated'):
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.save_dir = save_dir
        # Get device from model
        self.device = next(model.parameters()).device
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        # Create the folder to export images if not exists
        if not os.path.exists(os.path.join(self.save_dir)):
            os.makedirs(os.path.join(self.save_dir))

    def generate(self, iterations=600):
        """Generates class specific image

        Keyword Arguments:
            iterations {int} -- Total iterations for gradient ascent (default: {300})

        Returns:
            np.ndarray -- Final maximally activated class image
        """
        initial_learning_rate = 6
        # Add regularization parameters
        l2_reg = 1e-3  # L2 regularization strength
        blur_every = 4  # Apply blur every N iterations
        blur_sigma = 1.0  # Gaussian blur sigma
        
        for i in range(1, iterations):
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image, False)
            # Move to same device as model
            self.processed_image = self.processed_image.to(self.device)
            
            # Create a new tensor with requires_grad=True using clone().detach()
            self.processed_image = self.processed_image.clone().detach().requires_grad_(True)

            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            # Forward
            output = self.model(self.processed_image)
            # Handle tuple output (logits, features)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
                
            # Target specific class with L2 regularization
            class_loss = -logits[0, self.target_class] + l2_reg * torch.norm(self.processed_image)

            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            
            # Apply Gaussian blur periodically to reduce noise
            if blur_every > 0 and i % blur_every == 0:
                img_np = self.processed_image.detach().cpu().numpy()
                for c in range(3):
                    img_np[0, c] = cv2.GaussianBlur(img_np[0, c], (0, 0), blur_sigma)
                self.processed_image = torch.from_numpy(img_np).to(self.device)
                self.processed_image.requires_grad_(True)
                optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            
            # Recreate image
            self.created_image = recreate_image(self.processed_image.cpu())
            if i % iterations == 0 or i == iterations-1:
                # Save image
                im_path = os.path.join(self.save_dir, f'c_{self.target_class}_iter_{i}.png')
                save_image(self.created_image, im_path)

        return self.processed_image
    

