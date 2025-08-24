#!/usr/bin/env python3
"""
Simple launcher script for the Stable Diffusion ControlNet Pipeline
"""

import sys
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="🎨 Stable Diffusion ControlNet: Black & White to Photorealistic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --web                    # Launch web interface
  python run.py --single input.jpg       # Process single image
  python run.py --batch                  # Process all images in input_images/
  python run.py --examples               # Run basic examples
  python run.py --advanced               # Run advanced examples
        """
    )
    
    parser.add_argument("--web", action="store_true", 
                       help="Launch Gradio web interface")
    parser.add_argument("--single", type=str, 
                       help="Process a single image file")
    parser.add_argument("--batch", action="store_true", 
                       help="Process all images in input_images/ directory")
    parser.add_argument("--examples", action="store_true", 
                       help="Run basic usage examples")
    parser.add_argument("--advanced", action="store_true", 
                       help="Run advanced usage examples")
    parser.add_argument("--prompt", type=str, default="",
                       help="Text prompt for image generation")
    parser.add_argument("--palette", type=str, default="",
                       help="Color palette name (e.g., 'natural_warm', 'vintage_sepia', 'cinematic')")
    parser.add_argument("--output", type=str, 
                       help="Output file path (for single image)")
    parser.add_argument("--steps", type=int, default=20,
                       help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--seed", type=int, default=-1,
                       help="Random seed (-1 for random)")
    parser.add_argument("--list-palettes", action="store_true",
                       help="List all available color palettes")
    parser.add_argument("--photorealistic", action="store_true",
                       help="Use photorealistic settings for maximum preservation of original components")
    parser.add_argument("--extract-palette", type=str,
                       help="Extract color palette from an image file")
    parser.add_argument("--palette-name", type=str,
                       help="Custom name for extracted palette")
    parser.add_argument("--num-colors", type=int, default=5,
                       help="Number of colors to extract from image")
    parser.add_argument("--extraction-method", type=str, default="kmeans",
                       choices=["kmeans", "dominant"],
                       help="Method for extracting colors from image")
    parser.add_argument("--validate-image", type=str,
                       help="Validate if an image file is supported")
    parser.add_argument("--image-info", type=str,
                       help="Get detailed information about an image file")
    parser.add_argument("--supported-formats", action="store_true",
                       help="List all supported image formats")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help and launch web interface
    if len(sys.argv) == 1:
        print("🎨 Stable Diffusion ControlNet: Black & White to Photorealistic")
        print("=" * 60)
        print("No arguments provided. Launching web interface...")
        print("Use --help to see all available options.")
        print()
        args.web = True
    
    try:
        if args.extract_palette:
            print(f"🎨 Extracting color palette from: {args.extract_palette}")
            try:
                from color_palettes import extract_palette_from_image, add_palette_from_image
                
                # Extract palette
                palette_data = extract_palette_from_image(
                    args.extract_palette, 
                    args.num_colors, 
                    args.extraction_method
                )
                
                if palette_data:
                    print(f"✅ Palette extracted successfully!")
                    print(f"Name: {palette_data['name']}")
                    print(f"Colors: {', '.join(palette_data['colors'])}")
                    print(f"Method: {palette_data['extraction_method']}")
                    
                    # Add to custom palettes
                    success = add_palette_from_image(
                        args.extract_palette,
                        args.palette_name,
                        args.num_colors,
                        args.extraction_method
                    )
                    
                    if success:
                        palette_name = args.palette_name or palette_data['name']
                        print(f"✅ Palette '{palette_name}' added to custom palettes!")
                        print(f"💡 Use it with: python run.py --single image.jpg --palette {palette_name}")
                    else:
                        print("❌ Failed to add palette to custom palettes")
                else:
                    print("❌ Failed to extract palette from image")
                return
            except ImportError:
                print("❌ Color palettes module not available")
                return
            except Exception as e:
                print(f"❌ Error extracting palette: {e}")
                return
        
        if args.validate_image:
            print(f"🔍 Validating image: {args.validate_image}")
            try:
                from image_utils import validate_image_file
                is_valid, message = validate_image_file(args.validate_image)
                if is_valid:
                    print(f"✅ {message}")
                else:
                    print(f"❌ {message}")
                return
            except ImportError:
                print("❌ Image utilities module not available")
                return
            except Exception as e:
                print(f"❌ Error validating image: {e}")
                return
        
        if args.image_info:
            print(f"📊 Getting image info: {args.image_info}")
            try:
                from image_utils import ImageFormatHandler
                info = ImageFormatHandler.get_image_info(args.image_info)
                print(f"\n📋 Image Information:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
                return
            except ImportError:
                print("❌ Image utilities module not available")
                return
            except Exception as e:
                print(f"❌ Error getting image info: {e}")
                return
        
        if args.supported_formats:
            print("📋 Supported Image Formats:")
            try:
                from image_utils import ImageFormatHandler
                formats = ImageFormatHandler.get_supported_formats_list()
                print(f"\n🎨 Supported formats: {', '.join(formats)}")
                print(f"\n💡 Total supported formats: {len(formats)}")
                return
            except ImportError:
                print("❌ Image utilities module not available")
                return
            except Exception as e:
                print(f"❌ Error listing formats: {e}")
                return
        
        if args.list_palettes:
            print("🎨 Available Color Palettes:")
            try:
                from color_palettes import get_available_palettes, preview_palette
                palettes = get_available_palettes()
                
                print("\n📋 Predefined Palettes:")
                for palette in palettes["predefined"]:
                    preview = preview_palette(palette)
                    print(f"  • {preview['name']}: {preview['description']}")
                
                print("\n💾 Custom Palettes:")
                for palette in palettes["custom"]:
                    preview = preview_palette(palette)
                    print(f"  • {preview['name']}: {preview['description']}")
                
                print(f"\n💡 Usage: python run.py --single image.jpg --palette palette_name")
                return
            except ImportError:
                print("❌ Color palettes module not available")
                return
        
        if args.web:
            print("🚀 Launching web interface...")
            from gradio_interface import main as gradio_main
            gradio_main()
            
        elif args.single:
            print(f"📸 Processing single image: {args.single}")
            from stable_diffusion_controlnet import StableDiffusionControlNetPipeline_BWToColor
            
            pipeline = StableDiffusionControlNetPipeline_BWToColor()
            
            if args.photorealistic:
                print("🎯 Using photorealistic settings for maximum preservation...")
                output_path = pipeline.process_photorealistic_image(
                    input_path=args.single,
                    output_path=args.output,
                    preserve_structure=True,
                    enhance_details=True,
                    natural_colors=True,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    seed=args.seed if args.seed >= 0 else None
                )
            else:
                output_path = pipeline.process_image_file(
                    input_path=args.single,
                    output_path=args.output,
                    prompt=args.prompt,
                    color_palette=args.palette,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    seed=args.seed if args.seed >= 0 else None
                )
            print(f"✅ Colored image saved to: {output_path}")
            
        elif args.batch:
            print("📚 Processing batch of images...")
            from stable_diffusion_controlnet import StableDiffusionControlNetPipeline_BWToColor
            
            pipeline = StableDiffusionControlNetPipeline_BWToColor()
            
            output_paths = pipeline.process_batch(
                prompt=args.prompt,
                color_palette=args.palette,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed if args.seed >= 0 else None
            )
            
            if output_paths:
                print(f"✅ Successfully processed {len(output_paths)} images!")
                for path in output_paths:
                    print(f"   - {path}")
            else:
                print("❌ No images found in input_images/ directory")
                print("Please add some black and white images to the input_images/ folder")
            
        elif args.examples:
            print("📖 Running basic examples...")
            from examples.basic_usage import main as examples_main
            examples_main()
            
        elif args.advanced:
            print("🚀 Running advanced examples...")
            from examples.advanced_usage import main as advanced_main
            advanced_main()
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
