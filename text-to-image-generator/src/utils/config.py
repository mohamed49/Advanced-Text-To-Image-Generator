"""
Configuration settings for the text-to-image system
"""

import os


class Config:
    """Central configuration for the application"""
    
    # Model settings
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    MODE = "pretrained"  # "pretrained" or "custom"
    
    # Generation defaults
    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_WIDTH = 768
    DEFAULT_HEIGHT = 768
    
    # Quality settings
    HIGH_QUALITY_STEPS = 50
    MEDIUM_QUALITY_STEPS = 30
    LOW_QUALITY_STEPS = 20
    
    # Directories
    OUTPUT_DIR = "outputs"
    CACHE_DIR = "cache"
    
    # Device settings
    DEVICE = "cuda"  # Will be auto-detected
    
    # Flask settings
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 7860  # Changed to 7860 (better Lightning.ai support)
    FLASK_DEBUG = False
    
    # Prompt enhancement
    QUALITY_SUFFIX = ", high quality, detailed, vibrant colors, professional photography, 8k, sharp focus, good lighting"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


class ProductionConfig(Config):
    """Production-specific configuration"""
    FLASK_DEBUG = False
    DEFAULT_NUM_INFERENCE_STEPS = 30  # Faster for production


class DevelopmentConfig(Config):
    """Development-specific configuration"""
    FLASK_DEBUG = True
    DEFAULT_NUM_INFERENCE_STEPS = 20  # Faster for testing