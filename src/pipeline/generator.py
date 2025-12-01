"""
Main text-to-image generation system
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
from diffusers import UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

from ..models.unet import EnhancedTextConditionedUNet
from ..models.evaluator import ImageQualityEvaluator


class TextToImageGenerator:
    """
    Complete text-to-image generation system with:
    - Enhanced attention mechanisms
    - Quality evaluation
    - Pre-trained model support
    """
    
    def __init__(self, mode="pretrained", model_name="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = mode

        print("\n" + "="*70)
        print(f"🚀 Text-to-Image Generation System")
        print(f"Mode: {mode.upper()}")
        print(f"Device: {self.device}")
        print("="*70 + "\n")

        if mode == "pretrained":
            self._init_pretrained(model_name)
        elif mode == "custom":
            self._init_custom_enhanced()

        # Initialize evaluator
        self.evaluator = ImageQualityEvaluator(self.device)

    def _init_pretrained(self, model_name):
        """Initialize pre-trained Stable Diffusion with attention mechanisms"""
        print("📦 Loading Pre-trained Stable Diffusion...")
        print("   ✓ Built-in Cross-Attention between text and image")
        print("   ✓ Built-in Self-Attention in UNet layers")
        print("   ✓ Optimized attention mechanisms")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe = self.pipe.to(self.device)

        if self.device == "cuda":
            self.pipe.enable_attention_slicing()

        print("✅ Pre-trained model loaded with attention mechanisms\n")

    def _init_custom_enhanced(self):
        """Initialize custom model with enhanced attention"""
        print("🔨 Building Custom Model with Enhanced Attention...")

        # Text encoder
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_encoder = self.text_encoder.to(self.device)

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Enhanced UNet with attention
        self.model = EnhancedTextConditionedUNet()
        self.model = self.model.to(self.device)

        # Training components
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        self.loss_fn = nn.MSELoss()

        self.train_losses = []
        self.best_loss = float('inf')

        print("✅ Custom enhanced model initialized\n")

    def generate(
        self, 
        prompt, 
        num_inference_steps=50, 
        guidance_scale=7.5,
        width=512, 
        height=512, 
        seed=None
    ):
        """
        Generate image from text with attention mechanisms

        Args:
            prompt: Text description
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            width, height: Output dimensions
            seed: Random seed for reproducibility

        Returns:
            Generated PIL Image
        """
        if self.mode != "pretrained":
            raise ValueError("generate only works in 'pretrained' mode")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"🎨 Generating: '{prompt}'")
        print(f"   Using attention mechanisms to align text with visual elements...")

        with torch.autocast(self.device):
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            ).images[0]

        print(f"   ✅ Image generated successfully\n")

        return image

    def generate_batch(self, prompts, **kwargs):
        """Generate multiple images from multiple prompts"""
        images = []
        print(f"\n📸 Generating {len(prompts)} images...\n")

        for i, prompt in enumerate(prompts, 1):
            print(f"[{i}/{len(prompts)}] ", end="")
            image = self.generate(prompt, **kwargs)
            images.append((prompt, image))

        return images

    def evaluate_quality(self, generated_images, real_images=None):
        """
        Evaluate generated image quality

        Args:
            generated_images: List of PIL Images or tensor
            real_images: Optional real images for FID calculation

        Returns:
            Dictionary with metrics
        """
        print("\n📊 Evaluating Image Quality...")

        # Convert PIL to tensor if needed
        if isinstance(generated_images[0], Image.Image):
            gen_tensor = torch.stack([
                transforms.ToTensor()(img) for img in generated_images
            ])
        else:
            gen_tensor = generated_images

        results = {}

        # Calculate Inception Score
        try:
            is_mean, is_std = self.evaluator.calculate_inception_score(gen_tensor)
            results['inception_score'] = {
                'mean': float(is_mean),
                'std': float(is_std)
            }
            print(f"   ✓ Inception Score: {is_mean:.2f} ± {is_std:.2f}")
        except Exception as e:
            print(f"   ⚠ Inception Score calculation failed: {e}")

        # Calculate FID if real images provided
        if real_images is not None:
            try:
                if isinstance(real_images[0], Image.Image):
                    real_tensor = torch.stack([
                        transforms.ToTensor()(img) for img in real_images
                    ])
                else:
                    real_tensor = real_images

                fid = self.evaluator.calculate_fid(real_tensor, gen_tensor)
                results['fid'] = float(fid)
                print(f"   ✓ FID Score: {fid:.2f}")
            except Exception as e:
                print(f"   ⚠ FID calculation failed: {e}")

        print("✅ Quality evaluation completed\n")

        return results

    @staticmethod
    def display_images(images, figsize=(15, 5), save_path=None):
        """Display generated images with prompts"""
        if isinstance(images, tuple):
            images = [images]

        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, figsize=figsize)
        if n_images == 1:
            axes = [axes]

        for idx, (prompt, image) in enumerate(images):
            axes[idx].imshow(np.array(image))
            title = prompt[:50] + "..." if len(prompt) > 50 else prompt
            axes[idx].set_title(title, fontsize=10, wrap=True)
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved to {save_path}")

        plt.show()

    def save_generation_report(
        self, 
        prompts, 
        images, 
        metrics, 
        save_dir="outputs"
    ):
        """Save complete generation report with images and metrics"""
        os.makedirs(save_dir, exist_ok=True)

        # Save images
        for i, (prompt, image) in enumerate(zip(prompts, images)):
            image.save(os.path.join(save_dir, f"generated_{i+1}.png"))

        # Save report
        report_path = os.path.join(save_dir, "generation_report.txt")
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TEXT-TO-IMAGE GENERATION REPORT\n")
            f.write("="*70 + "\n\n")

            f.write("PROMPTS:\n")
            for i, prompt in enumerate(prompts, 1):
                f.write(f"{i}. {prompt}\n")

            f.write("\n" + "="*70 + "\n")
            f.write("QUALITY METRICS:\n")
            f.write("="*70 + "\n")

            if 'inception_score' in metrics:
                is_data = metrics['inception_score']
                f.write(
                    f"Inception Score: {is_data['mean']:.2f} ± {is_data['std']:.2f}\n"
                )
                f.write("  (Higher is better - measures quality and diversity)\n\n")

            if 'fid' in metrics:
                f.write(f"FID Score: {metrics['fid']:.2f}\n")
                f.write(
                    "  (Lower is better - measures similarity to real images)\n\n"
                )

            f.write("\n" + "="*70 + "\n")
            f.write("ATTENTION MECHANISMS USED:\n")
            f.write("="*70 + "\n")
            f.write("✓ Cross-Attention: Aligns text features with image features\n")
            f.write("✓ Self-Attention: Captures long-range dependencies\n")
            f.write("✓ Multi-Head Attention: Multiple attention perspectives\n")

        print(f"📄 Report saved to {report_path}")