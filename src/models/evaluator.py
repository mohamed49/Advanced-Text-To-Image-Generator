"""
Image quality evaluation metrics
Includes Inception Score and FID calculation
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg


class ImageQualityEvaluator:
    """
    Evaluates generated image quality using:
    - Inception Score (IS)
    - Fréchet Inception Distance (FID)
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        print("📊 Initializing Image Quality Evaluator...")

        # Load Inception V3 for evaluation
        self.inception_model = inception_v3(pretrained=True, transform_input=False)
        self.inception_model.fc = nn.Identity()  # Remove final layer
        self.inception_model = self.inception_model.to(device)
        self.inception_model.eval()

        print("✅ Evaluator ready")

    def preprocess_images(self, images):
        """Preprocess images for Inception model"""
        if isinstance(images, list):
            images = torch.stack([transforms.ToTensor()(img) for img in images])

        # Resize to 299x299 for Inception
        images = torch.nn.functional.interpolate(
            images, size=(299, 299), mode='bilinear', align_corners=False
        )

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std

        return images

    def get_inception_features(self, images):
        """Extract features using Inception model"""
        with torch.no_grad():
            images = self.preprocess_images(images).to(self.device)
            features = self.inception_model(images)
        return features.cpu().numpy()

    def calculate_inception_score(self, images, splits=10):
        """
        Calculate Inception Score (IS)
        Higher is better (measures quality and diversity)
        
        Args:
            images: Generated images
            splits: Number of splits for calculation
        
        Returns:
            mean, std: IS mean and standard deviation
        """
        features = self.get_inception_features(images)

        # Calculate predictions
        preds = torch.nn.functional.softmax(torch.tensor(features), dim=1).numpy()

        # Split into groups
        split_scores = []
        for k in range(splits):
            part = preds[
                k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :
            ]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(np.sum(pyx * np.log(pyx / py)))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

    def calculate_fid(self, real_images, generated_images):
        """
        Calculate Fréchet Inception Distance (FID)
        Lower is better (measures similarity to real images)
        
        Args:
            real_images: Real reference images
            generated_images: Generated images
        
        Returns:
            fid: FID score
        """
        # Get features
        real_features = self.get_inception_features(real_images)
        gen_features = self.get_inception_features(generated_images)

        # Calculate mean and covariance
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

        # Calculate FID
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

        return fid