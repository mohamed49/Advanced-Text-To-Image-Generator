"""
Image processing utilities
"""

import io
import base64
from PIL import Image
import numpy as np


def pil_to_base64(image):
    """
    Convert PIL Image to base64 string for web display
    
    Args:
        image: PIL Image
    
    Returns:
        Base64 encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def base64_to_pil(base64_string):
    """
    Convert base64 string to PIL Image
    
    Args:
        base64_string: Base64 encoded image string
    
    Returns:
        PIL Image
    """
    img_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(img_data))
    return image


def resize_image(image, width, height, maintain_aspect=True):
    """
    Resize image to specified dimensions
    
    Args:
        image: PIL Image
        width: Target width
        height: Target height
        maintain_aspect: Whether to maintain aspect ratio
    
    Returns:
        Resized PIL Image
    """
    if maintain_aspect:
        image.thumbnail((width, height), Image.LANCZOS)
    else:
        image = image.resize((width, height), Image.LANCZOS)
    return image


def enhance_prompt(prompt, quality_level="high"):
    """
    Enhance user prompt with quality descriptors
    
    Args:
        prompt: Original user prompt
        quality_level: "high", "medium", or "low"
    
    Returns:
        Enhanced prompt string
    """
    enhancements = {
        "high": ", high quality, detailed, vibrant colors, professional photography, 8k, sharp focus, good lighting, masterpiece",
        "medium": ", good quality, detailed, vibrant colors, sharp focus",
        "low": ", good quality"
    }
    
    enhancement = enhancements.get(quality_level, enhancements["medium"])
    return f"{prompt}{enhancement}"


def create_image_grid(images, grid_size=None):
    """
    Create a grid of images
    
    Args:
        images: List of PIL Images
        grid_size: Tuple (rows, cols) or None for auto
    
    Returns:
        PIL Image grid
    """
    if not images:
        return None
    
    n_images = len(images)
    
    if grid_size is None:
        # Auto-calculate grid size
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    # Get dimensions from first image
    img_width, img_height = images[0].size
    
    # Create blank grid
    grid_width = cols * img_width
    grid_height = rows * img_height
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))
    
    return grid