#!/bin/bash

# Lightning.ai Setup Script for Text-to-Image Generator
# This script automates the setup process on Lightning.ai

echo "=========================================="
echo "🚀 Lightning.ai Setup"
echo "🎨 Text-to-Image Generator"
echo "=========================================="
echo ""

# Step 1: Check Python version
echo "📌 Step 1: Checking Python version..."
python --version
echo ""

# Step 2: Upgrade pip
echo "📌 Step 2: Upgrading pip..."
pip install --upgrade pip
echo ""

# Step 3: Install PyTorch with CUDA support
echo "📌 Step 3: Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo ""

# Step 4: Fix NumPy/scikit-learn compatibility
echo "📌 Step 4: Installing NumPy and scikit-learn with compatible versions..."
pip install numpy==1.26.4 scikit-learn==1.5.0
echo ""

# Step 5: Install other requirements
echo "📌 Step 5: Installing remaining dependencies..."
pip install -r requirements.txt
echo ""

# Step 5: Create necessary directories
echo "📌 Step 5: Creating directories..."
mkdir -p outputs
mkdir -p cache
echo "✓ Created outputs/ and cache/ directories"
echo ""

# Step 6: Download and cache models
echo "📌 Step 6: Pre-downloading models (this may take a few minutes)..."
python -c "
from diffusers import StableDiffusionPipeline
import torch

print('Downloading Stable Diffusion model...')
pipe = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=torch.float16,
    safety_checker=None
)
print('✓ Model downloaded and cached')
"
echo ""

# Step 7: Verify CUDA availability
echo "📌 Step 7: Verifying CUDA..."
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
else:
    print('⚠️  WARNING: CUDA not available, using CPU (will be slower)')
"
echo ""

# Step 8: Test imports
echo "📌 Step 8: Testing imports..."
python -c "
import torch
import transformers
import diffusers
from flask import Flask
print('✓ All imports successful')
"
echo ""

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: python app.py"
echo "2. Open the provided URL in your browser"
echo "3. Start generating images!"
echo ""
echo "For help, see README.md"
echo "=========================================="