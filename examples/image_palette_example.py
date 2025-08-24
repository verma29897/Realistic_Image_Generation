#!/usr/bin/env python3
"""
Image Palette Extraction Example
Demonstrates extracting color palettes from images and using them for colorization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from color_palettes import extract_palette_from_image, add_palette_from_image, get_available_palettes


def main():
    """Demonstrate image palette extraction and usage."""
    print("🎨 Image Palette Extraction Example")
    print("=" * 50)
    
    # Example image path (you can change this to any image)
    example_image = "input_images/test.jpg"  # or any other image
    
    print(f"\n📸 Extracting palette from: {example_image}")
    
    try:
        # Method 1: Extract palette without saving
        print("\n🔍 Method 1: Extract and preview palette")
        palette_data = extract_palette_from_image(
            image_path=example_image,
            num_colors=5,
            method="kmeans"
        )
        
        if palette_data:
            print(f"✅ Palette extracted successfully!")
            print(f"Name: {palette_data['name']}")
            print(f"Description: {palette_data['description']}")
            print(f"Colors: {', '.join(palette_data['colors'])}")
            print(f"Method: {palette_data['extraction_method']}")
        else:
            print("❌ Failed to extract palette")
            return
        
        # Method 2: Extract and save as custom palette
        print("\n💾 Method 2: Extract and save as custom palette")
        success = add_palette_from_image(
            image_path=example_image,
            palette_name="my_custom_palette",
            num_colors=6,
            method="dominant"
        )
        
        if success:
            print("✅ Custom palette saved successfully!")
        else:
            print("❌ Failed to save custom palette")
        
        # Method 3: Show all available palettes
        print("\n📋 Method 3: List all available palettes")
        palettes = get_available_palettes()
        
        print("\nPredefined Palettes:")
        for palette in palettes["predefined"][:5]:  # Show first 5
            print(f"  • {palette}")
        
        print("\nCustom Palettes:")
        for palette in palettes["custom"]:
            print(f"  • {palette}")
        
        # Method 4: Use extracted palette for colorization
        print("\n🎨 Method 4: Use extracted palette for colorization")
        print("You can now use the extracted palette for colorizing images:")
        print(f"python run.py --single input_images/test.jpg --palette {palette_data['name']}")
        print("python run.py --single input_images/test.jpg --palette my_custom_palette")
        
        # Method 5: Different extraction methods
        print("\n🔬 Method 5: Compare extraction methods")
        methods = ["kmeans", "dominant"]
        
        for method in methods:
            print(f"\n{method.upper()} method:")
            method_palette = extract_palette_from_image(
                image_path=example_image,
                num_colors=4,
                method=method
            )
            if method_palette:
                print(f"  Colors: {', '.join(method_palette['colors'])}")
        
    except FileNotFoundError:
        print(f"❌ Image not found at {example_image}")
        print("Please place an image in the input_images directory")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
