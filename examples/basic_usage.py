#!/usr/bin/env python3
"""
Basic usage examples for the Stable Diffusion ControlNet Black & White to Color Pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_diffusion_controlnet import StableDiffusionControlNetPipeline_BWToColor
from PIL import Image
import requests
from io import BytesIO


def download_sample_image(url: str, save_path: str):
    """Download a sample black and white image for testing."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
        print(f"Downloaded sample image to: {save_path}")
        return save_path
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def example_single_image():
    """Example: Process a single black and white image."""
    print("=== Single Image Processing Example ===")
    
    # Initialize the pipeline
    pipeline = StableDiffusionControlNetPipeline_BWToColor()
    
    # You can either use your own image or download a sample
    # For this example, let's assume you have an image in the input directory
    input_path = "input_images/sample_bw.jpg"
    
    # Create a sample prompt
    prompt = "beautiful portrait of a woman, natural skin tones, soft lighting, warm colors, professional photography"
    
    try:
        # Process the image
        output_path = pipeline.process_image_file(
            input_path=input_path,
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=42
        )
        print(f"✅ Success! Colored image saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"❌ Image not found at {input_path}")
        print("Please place a black and white image in the input_images directory or use the batch processing example")
    except Exception as e:
        print(f"❌ Error processing image: {e}")


def example_batch_processing():
    """Example: Process multiple images in batch."""
    print("\n=== Batch Processing Example ===")
    
    # Initialize the pipeline
    pipeline = StableDiffusionControlNetPipeline_BWToColor()
    
    # Process all images in the input directory
    prompt = "photorealistic, vibrant colors, natural lighting, high quality, professional photography"
    
    try:
        output_paths = pipeline.process_batch(
            prompt=prompt,
            num_inference_steps=15,  # Faster for batch processing
            guidance_scale=7.0,
            seed=42
        )
        
        if output_paths:
            print(f"✅ Successfully processed {len(output_paths)} images!")
            for path in output_paths:
                print(f"   - {path}")
        else:
            print("❌ No images found to process")
            print("Please add some black and white images to the input_images directory")
            
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")


def example_custom_settings():
    """Example: Process with custom settings for different scenarios."""
    print("\n=== Custom Settings Examples ===")
    
    # Initialize the pipeline
    pipeline = StableDiffusionControlNetPipeline_BWToColor()
    
    # Example scenarios with different settings
    scenarios = [
        {
            "name": "High Quality Portrait",
            "prompt": "professional headshot, natural skin tones, studio lighting, sharp focus, 8k quality",
            "negative_prompt": "blurry, low quality, cartoon, anime, oversaturated",
            "num_inference_steps": 30,
            "guidance_scale": 8.0,
            "controlnet_conditioning_scale": 1.2
        },
        {
            "name": "Vintage Photo Restoration",
            "prompt": "restored vintage photograph, sepia tones converted to natural colors, film photography, nostalgic",
            "negative_prompt": "modern, digital, oversaturated, artificial",
            "num_inference_steps": 25,
            "guidance_scale": 7.0,
            "controlnet_conditioning_scale": 1.0
        },
        {
            "name": "Artistic Creative",
            "prompt": "artistic interpretation, vibrant colors, creative lighting, painterly style, cinematic",
            "negative_prompt": "boring, dull, monochrome, low contrast",
            "num_inference_steps": 20,
            "guidance_scale": 6.0,
            "controlnet_conditioning_scale": 0.8
        }
    ]
    
    input_path = "input_images/sample_bw.jpg"  # Replace with your image
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['name']} ---")
        
        try:
            output_path = f"output_images/example_{i}_{scenario['name'].lower().replace(' ', '_')}.png"
            
            # Generate with custom settings
            colored_images = pipeline.generate_colored_image(
                bw_image=input_path,
                prompt=scenario['prompt'],
                negative_prompt=scenario['negative_prompt'],
                num_inference_steps=scenario['num_inference_steps'],
                guidance_scale=scenario['guidance_scale'],
                controlnet_conditioning_scale=scenario['controlnet_conditioning_scale'],
                seed=42 + i  # Different seed for each scenario
            )
            
            # Save the result
            colored_images[0].save(output_path, quality=95)
            print(f"✅ Generated: {output_path}")
            
        except FileNotFoundError:
            print(f"❌ Input image not found: {input_path}")
            break
        except Exception as e:
            print(f"❌ Error in scenario {i}: {e}")


def example_multiple_variations():
    """Example: Generate multiple variations of the same image."""
    print("\n=== Multiple Variations Example ===")
    
    # Initialize the pipeline
    pipeline = StableDiffusionControlNetPipeline_BWToColor()
    
    input_path = "input_images/sample_bw.jpg"  # Replace with your image
    base_prompt = "beautiful portrait, natural colors, professional photography"
    
    # Generate multiple variations with different seeds
    num_variations = 3
    
    try:
        for i in range(num_variations):
            print(f"Generating variation {i+1}/{num_variations}...")
            
            colored_images = pipeline.generate_colored_image(
                bw_image=input_path,
                prompt=base_prompt,
                seed=42 + i * 1000,  # Different seed for each variation
                num_inference_steps=20
            )
            
            output_path = f"output_images/variation_{i+1}.png"
            colored_images[0].save(output_path, quality=95)
            print(f"✅ Saved variation {i+1}: {output_path}")
        
        print(f"✅ Generated {num_variations} variations successfully!")
        
    except FileNotFoundError:
        print(f"❌ Input image not found: {input_path}")
    except Exception as e:
        print(f"❌ Error generating variations: {e}")


def setup_sample_images():
    """Download some sample black and white images for testing."""
    print("=== Setting up sample images ===")
    
    # Create input directory
    os.makedirs("input_images", exist_ok=True)
    
    # Sample black and white images (you would replace these with actual URLs)
    # Note: These are placeholder URLs - you would need to use actual B&W image URLs
    sample_images = [
        {
            "url": "https://example.com/sample_bw_portrait.jpg",  # Replace with actual URL
            "filename": "input_images/sample_portrait.jpg"
        },
        {
            "url": "https://example.com/sample_bw_landscape.jpg",  # Replace with actual URL
            "filename": "input_images/sample_landscape.jpg"
        }
    ]
    
    print("Note: Please manually place some black and white images in the 'input_images' directory")
    print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp")
    print("For best results, use high-contrast black and white images")


if __name__ == "__main__":
    print("🎨 Stable Diffusion ControlNet Black & White to Color - Examples")
    print("=" * 60)
    
    # Setup sample images directory
    setup_sample_images()
    
    # Run examples (uncomment the ones you want to try)
    
    # Example 1: Single image processing
    # example_single_image()
    
    # Example 2: Batch processing
    # example_batch_processing()
    
    # Example 3: Custom settings for different scenarios
    # example_custom_settings()
    
    # Example 4: Multiple variations
    # example_multiple_variations()
    
    print("\n💡 To run the examples:")
    print("1. Place some black and white images in the 'input_images' directory")
    print("2. Uncomment the example functions you want to try")
    print("3. Run this script again")
    print("\n🚀 Or launch the web interface with: python gradio_interface.py")
