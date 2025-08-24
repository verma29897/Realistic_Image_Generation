#!/usr/bin/env python3
"""
Image Utilities for Enhanced Format Support
Provides comprehensive image format handling and processing utilities
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Union
from PIL import Image, ImageOps
import numpy as np
import cv2


class ImageFormatHandler:
    """Handles various image formats with enhanced support."""
    
    # Supported image formats
    SUPPORTED_FORMATS = {
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
        '.webp', '.gif', '.ico', '.ppm', '.pgm', '.pbm'
    }
    
    # Common image modes and their handling
    MODE_HANDLING = {
        'RGB': 'direct',
        'RGBA': 'alpha_to_white',
        'LA': 'grayscale_to_rgb',
        'L': 'grayscale_to_rgb',
        'P': 'palette_to_rgb',
        'CMYK': 'cmyk_to_rgb',
        'YCbCr': 'ycbcr_to_rgb',
        'HSV': 'hsv_to_rgb',
        'LAB': 'lab_to_rgb'
    }
    
    @classmethod
    def is_supported_format(cls, file_path: str) -> bool:
        """Check if the file format is supported."""
        return Path(file_path).suffix.lower() in cls.SUPPORTED_FORMATS
    
    @classmethod
    def get_supported_formats_list(cls) -> List[str]:
        """Get list of supported file extensions."""
        return sorted(list(cls.SUPPORTED_FORMATS))
    
    @classmethod
    def load_image(cls, image_path: str, target_mode: str = 'RGB') -> Optional[Image.Image]:
        """
        Load image with enhanced format support.
        
        Args:
            image_path: Path to the image file
            target_mode: Target color mode (default: 'RGB')
            
        Returns:
            PIL Image object or None if loading fails
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"Error: File not found: {image_path}")
                return None
            
            # Check format support
            file_ext = Path(image_path).suffix.lower()
            if file_ext not in cls.SUPPORTED_FORMATS:
                print(f"Warning: File format {file_ext} may not be fully supported.")
            
            # Load image
            image = Image.open(image_path)
            
            # Handle different image modes
            image = cls._convert_image_mode(image, target_mode)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    @classmethod
    def _convert_image_mode(cls, image: Image.Image, target_mode: str = 'RGB') -> Image.Image:
        """
        Convert image to target mode with proper handling.
        
        Args:
            image: PIL Image object
            target_mode: Target color mode
            
        Returns:
            Converted PIL Image object
        """
        current_mode = image.mode
        
        if current_mode == target_mode:
            return image
        
        # Handle specific mode conversions
        if current_mode == 'RGBA':
            if target_mode == 'RGB':
                # Convert RGBA to RGB with white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                return background
            else:
                return image.convert(target_mode)
        
        elif current_mode in ['LA', 'L']:
            if target_mode == 'RGB':
                # Convert grayscale to RGB
                return image.convert('RGB')
            else:
                return image.convert(target_mode)
        
        elif current_mode == 'P':
            # Handle palette mode
            if target_mode == 'RGB':
                return image.convert('RGB')
            else:
                return image.convert(target_mode)
        
        elif current_mode in ['CMYK', 'YCbCr', 'HSV', 'LAB']:
            # Convert to RGB first, then to target mode
            rgb_image = image.convert('RGB')
            if target_mode != 'RGB':
                return rgb_image.convert(target_mode)
            return rgb_image
        
        else:
            # Default conversion
            return image.convert(target_mode)
    
    @classmethod
    def save_image(cls, image: Image.Image, output_path: str, format: str = None, quality: int = 95) -> bool:
        """
        Save image with format detection and optimization.
        
        Args:
            image: PIL Image object to save
            output_path: Output file path
            format: Output format (auto-detected from extension if None)
            quality: JPEG quality (1-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Auto-detect format from extension
            if format is None:
                format = Path(output_path).suffix.lower().lstrip('.')
                if format == 'jpg':
                    format = 'JPEG'
                elif format == 'tif':
                    format = 'TIFF'
                else:
                    format = format.upper()
            
            # Prepare save parameters
            save_kwargs = {}
            
            if format == 'JPEG':
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            elif format == 'PNG':
                save_kwargs['optimize'] = True
            elif format == 'WEBP':
                save_kwargs['quality'] = quality
                save_kwargs['method'] = 6  # Best compression
            
            # Save image
            image.save(output_path, format=format, **save_kwargs)
            return True
            
        except Exception as e:
            print(f"Error saving image to {output_path}: {e}")
            return False
    
    @classmethod
    def get_image_info(cls, image_path: str) -> dict:
        """
        Get comprehensive information about an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image information
        """
        try:
            image = Image.open(image_path)
            
            info = {
                'filename': Path(image_path).name,
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height,
                'aspect_ratio': image.width / image.height,
                'file_size_mb': os.path.getsize(image_path) / (1024 * 1024),
                'is_supported': cls.is_supported_format(image_path)
            }
            
            # Add color information for RGB images
            if image.mode in ['RGB', 'RGBA']:
                img_array = np.array(image)
                info['mean_color'] = tuple(np.mean(img_array, axis=(0, 1)).astype(int))
                info['std_color'] = tuple(np.std(img_array, axis=(0, 1)).astype(int))
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    @classmethod
    def batch_convert_formats(cls, input_dir: str, output_dir: str, 
                            target_format: str = 'JPEG', quality: int = 95) -> List[str]:
        """
        Batch convert images to a specific format.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            target_format: Target format (JPEG, PNG, etc.)
            quality: JPEG quality (1-100)
            
        Returns:
            List of successfully converted file paths
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        converted_files = []
        
        for file_path in input_path.iterdir():
            if file_path.is_file() and cls.is_supported_format(str(file_path)):
                try:
                    # Load image
                    image = cls.load_image(str(file_path))
                    if image is None:
                        continue
                    
                    # Generate output path
                    output_file = output_path / f"{file_path.stem}.{target_format.lower()}"
                    
                    # Save with new format
                    if cls.save_image(image, str(output_file), target_format, quality):
                        converted_files.append(str(output_file))
                        print(f"Converted: {file_path.name} -> {output_file.name}")
                    
                except Exception as e:
                    print(f"Error converting {file_path.name}: {e}")
        
        return converted_files


def validate_image_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate if a file is a supported image format.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    if not ImageFormatHandler.is_supported_format(file_path):
        return False, f"Unsupported format: {Path(file_path).suffix}"
    
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True, "Valid image file"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def get_image_dimensions(file_path: str) -> Optional[Tuple[int, int]]:
    """
    Get image dimensions without loading the entire image.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Tuple of (width, height) or None if error
    """
    try:
        with Image.open(file_path) as img:
            return img.size
    except Exception:
        return None


if __name__ == "__main__":
    # Example usage
    print("🖼️ Image Format Support Utilities")
    print("=" * 40)
    
    print(f"\n📋 Supported formats: {', '.join(ImageFormatHandler.get_supported_formats_list())}")
    
    # Example: Check if a file is supported
    test_file = "input_images/test.jpg"
    if os.path.exists(test_file):
        is_valid, message = validate_image_file(test_file)
        print(f"\n✅ File validation: {message}")
        
        # Get image info
        info = ImageFormatHandler.get_image_info(test_file)
        print(f"\n📊 Image info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print(f"\n❌ Test file not found: {test_file}")
