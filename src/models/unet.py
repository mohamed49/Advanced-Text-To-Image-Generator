"""
Enhanced UNet model with attention mechanisms
"""

import torch.nn as nn
from diffusers import UNet2DConditionModel
from .attention import SelfAttentionLayer, CrossAttentionLayer


class EnhancedTextConditionedUNet(nn.Module):
    """
    Enhanced UNet with Multi-Head Attention Mechanisms
    - Self-attention for better feature extraction
    - Cross-attention for text-image alignment
    """
    
    def __init__(self, text_embed_dim=768, image_channels=3):
        super(EnhancedTextConditionedUNet, self).__init__()

        # Base UNet with cross-attention
        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=image_channels,
            out_channels=image_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=[
                "DownBlock2D", 
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D", 
                "DownBlock2D",
            ],
            up_block_types=[
                "UpBlock2D", 
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D", 
                "UpBlock2D",
            ],
            cross_attention_dim=text_embed_dim,
        )

        # Additional attention layers
        self.self_attention = SelfAttentionLayer(text_embed_dim, num_heads=8)
        self.cross_attention = CrossAttentionLayer(text_embed_dim, num_heads=8)

        print("✅ Enhanced UNet with Multi-Head Attention initialized")

    def forward(self, noisy_images, timesteps, text_embeddings):
        """
        Forward pass through the UNet
        
        Args:
            noisy_images: Noisy input images
            timesteps: Diffusion timesteps
            text_embeddings: Encoded text embeddings
        
        Returns:
            Denoised output
        """
        # Apply self-attention to text embeddings
        text_enhanced, _ = self.self_attention(text_embeddings)

        # Forward through UNet with enhanced text
        output = self.unet(
            noisy_images, 
            timesteps, 
            encoder_hidden_states=text_enhanced
        ).sample

        return output