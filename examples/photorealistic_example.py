#!/usr/bin/env python3
"""
Photorealistic Black & White to Color Conversion Example
Demonstrates maximum preservation of original photo components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_diffusion_controlnet import StableDiffusionControlNetPipeline_BWToColor


def main():
    """Demonstrate photorealistic colorization with maximum preservation."""
    print("🎯 Photorealistic Black & White to Color Conversion")
    print("=" * 60)
    
    # Initialize the pipeline
    pipeline = StableDiffusionControlNetPipeline_BWToColor()
    
    # Example input image path
    input_path = "input_images/test.jpg"
    
    print(f"\n📸 Processing: {input_path}")
    print("🎯 Using photorealistic settings for maximum preservation...")
    
    try:
        # Process with photorealistic settings
        output_path = pipeline.process_photorealistic_image(
            input_path=input_path,
            preserve_structure=True,      # Preserve original composition
            enhance_details=True,         # Enhance fine details
            natural_colors=True,          # Use natural color palette
            num_inference_steps=30,       # More steps for better quality
            guidance_scale=9.0,           # Higher guidance for faithfulness
            controlnet_conditioning_scale=1.3,  # Stronger control
            seed=42                       # Reproducible results
        )
        
        print(f"✅ Photorealistic image saved to: {output_path}")
        print("\n🎨 Key Features Applied:")
        print("  • Enhanced preprocessing with CLAHE contrast enhancement")
        print("  • Adaptive Canny edge detection for better structure preservation")
        print("  • Aspect ratio preserving resize")
        print("  • Specialized photorealistic prompts")
        print("  • Higher inference steps (30) for better quality")
        print("  • Stronger ControlNet conditioning (1.3) for structure preservation")
        print("  • Higher guidance scale (9.0) for faithfulness")
        
    except FileNotFoundError:
        print(f"❌ Image not found at {input_path}")
        print("Please place a black and white image in the input_images directory")
    except Exception as e:
        print(f"❌ Error processing image: {e}")


if __name__ == "__main__":
    main()
