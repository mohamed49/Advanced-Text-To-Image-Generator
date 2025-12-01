"""
Attention mechanisms for text-to-image generation
Includes Cross-Attention and Self-Attention layers
"""

import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """Enhanced Cross-Attention mechanism for text-image alignment"""
    
    def __init__(self, embed_dim=768, num_heads=8):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, query, key_value):
        """
        Args:
            query: Image features [batch, seq_len, embed_dim]
            key_value: Text features [batch, seq_len, embed_dim]
        
        Returns:
            output: Enhanced features
            attn_weights: Attention weights
        """
        # Cross-attention: query from image, key/value from text
        attn_output, attn_weights = self.multihead_attn(
            query, key_value, key_value
        )
        query = self.norm1(query + attn_output)

        # Feed-forward
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)

        return output, attn_weights


class SelfAttentionLayer(nn.Module):
    """Self-Attention mechanism for better feature representation"""
    
    def __init__(self, embed_dim=768, num_heads=8):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: Input features [batch, seq_len, embed_dim]
        
        Returns:
            output: Self-attended features
            attn_weights: Attention weights
        """
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        output = self.norm(x + attn_output)
        return output, attn_weights