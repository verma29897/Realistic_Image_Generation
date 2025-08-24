#!/usr/bin/env python3
"""
Enhanced Image Format Support Example
Demonstrates comprehensive image format handling and processing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_utils import ImageFormatHandler, validate_image_file
from stable_diffusion_controlnet import StableDiffusionControlNetPipeline_BWToColor


def main():
    """Demonstrate enhanced image format support."""
    print("🖼️ Enhanced Image Format Support Example")
    print("=" * 50)
    
    # Show supported formats
    print(f"\n📋 Supported Image Formats:")
    formats = ImageFormatHandler.get_supported_formats_list()
    print(f"  {', '.join(formats)}")
    print(f"  Total: {len(formats)} formats supported")
    
    # Example image paths (you can test with different formats)
    test_images = [
        "input_images/test.jpg",
        # Add more test images with different formats here
        # "input_images/test.png",
        # "input_images/test.bmp",
        # "input_images/test.tiff",
        # "input_images/test.webp"
    ]
    
    print(f"\n🔍 Testing Image Validation:")
    for image_path in test_images:
        if os.path.exists(image_path):
            is_valid, message = validate_image_file(image_path)
            print(f"  {image_path}: {'✅' if is_valid else '❌'} {message}")
        else:
            print(f"  {image_path}: ❌ File not found")
    
    print(f"\n📊 Getting Image Information:")
    for image_path in test_images:
        if os.path.exists(image_path):
            info = ImageFormatHandler.get_image_info(image_path)
            print(f"\n  📸 {info['filename']}:")
            print(f"    Format: {info['format']}")
            print(f"    Mode: {info['mode']}")
            print(f"    Size: {info['width']}x{info['height']}")
            print(f"    Aspect Ratio: {info['aspect_ratio']:.2f}")
            print(f"    File Size: {info['file_size_mb']:.2f} MB")
            print(f"    Supported: {info['is_supported']}")
    
    print(f"\n🎨 Testing Colorization with Different Formats:")
    
    # Initialize pipeline
    try:
        pipeline = StableDiffusionControlNetPipeline_BWToColor()
        
        for image_path in test_images:
            if os.path.exists(image_path):
                print(f"\n  Processing: {image_path}")
                
                # Test photorealistic processing
                try:
                    output_path = pipeline.process_photorealistic_image(
                        input_path=image_path,
                        preserve_structure=True,
                        enhance_details=True,
                        natural_colors=True,
                        num_inference_steps=20,  # Faster for testing
                        guidance_scale=8.0
                    )
                    print(f"    ✅ Photorealistic output: {output_path}")
                except Exception as e:
                    print(f"    ❌ Error processing: {e}")
            else:
                print(f"  Skipping: {image_path} (not found)")
                
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
    
    print(f"\n💡 Usage Examples:")
    print(f"  # Validate any image format")
    print(f"  python run.py --validate-image your_image.png")
    print(f"  python run.py --validate-image your_image.tiff")
    print(f"  python run.py --validate-image your_image.webp")
    
    print(f"\n  # Get detailed image information")
    print(f"  python run.py --image-info your_image.jpg")
    
    print(f"\n  # List all supported formats")
    print(f"  python run.py --supported-formats")
    
    print(f"\n  # Process any supported format")
    print(f"  python run.py --single your_image.png --photorealistic")
    print(f"  python run.py --single your_image.tiff --photorealistic")
    print(f"  python run.py --single your_image.webp --photorealistic")
    
    print(f"\n  # Extract palette from any format")
    print(f"  python run.py --extract-palette reference_image.png --palette-name 'my_palette'")
    print(f"  python run.py --extract-palette reference_image.tiff --palette-name 'my_palette'")


if __name__ == "__main__":
    main()
