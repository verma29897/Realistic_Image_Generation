#!/usr/bin/env python3
"""
Advanced usage examples for the Stable Diffusion ControlNet Pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_diffusion_controlnet import StableDiffusionControlNetPipeline_BWToColor
from PIL import Image, ImageEnhance, ImageFilter
import torch
import time
import json


class AdvancedPipeline(StableDiffusionControlNetPipeline_BWToColor):
    """Extended pipeline with advanced features."""
    
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        self.generation_history = []
    
    def preprocess_with_enhancement(self, image, enhance_contrast=1.2, enhance_sharpness=1.1):
        """Preprocess image with enhancement options."""
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast and sharpness for better edge detection
        if enhance_contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(enhance_contrast)
        
        if enhance_sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(enhance_sharpness)
        
        return self.preprocess_bw_image(image)
    
    def generate_with_style_transfer(self, bw_image, style_prompt, content_weight=0.7):
        """Generate image with style transfer approach."""
        # Create style-focused prompts
        style_positive = f"in the style of {style_prompt}, artistic interpretation, {style_prompt} colors and lighting"
        style_negative = "realistic photography, documentary style, plain, boring"
        
        return self.generate_colored_image(
            bw_image=bw_image,
            prompt=style_positive,
            negative_prompt=style_negative,
            controlnet_conditioning_scale=content_weight,
            guidance_scale=8.0,
            num_inference_steps=25
        )
    
    def generate_progressive_refinement(self, bw_image, base_prompt, refinement_steps=3):
        """Generate image with progressive refinement."""
        results = []
        current_prompt = base_prompt
        
        for step in range(refinement_steps):
            print(f"Refinement step {step + 1}/{refinement_steps}")
            
            # Gradually increase quality and detail in prompts
            if step == 0:
                step_prompt = f"{current_prompt}, rough coloring, initial colors"
                steps = 15
            elif step == 1:
                step_prompt = f"{current_prompt}, improved colors, more detail, refined"
                steps = 20
            else:
                step_prompt = f"{current_prompt}, perfect colors, high detail, professional, masterpiece"
                steps = 25
            
            images = self.generate_colored_image(
                bw_image=bw_image,
                prompt=step_prompt,
                num_inference_steps=steps,
                seed=42 + step,
                controlnet_conditioning_scale=1.0 + (step * 0.1)
            )
            
            results.append(images[0])
        
        return results
    
    def benchmark_generation(self, bw_image, prompt, iterations=5):
        """Benchmark generation performance."""
        print(f"Benchmarking generation performance ({iterations} iterations)...")
        
        times = []
        for i in range(iterations):
            start_time = time.time()
            
            images = self.generate_colored_image(
                bw_image=bw_image,
                prompt=prompt,
                seed=42 + i,
                num_inference_steps=20
            )
            
            generation_time = time.time() - start_time
            times.append(generation_time)
            print(f"Iteration {i+1}: {generation_time:.2f}s")
        
        avg_time = sum(times) / len(times)
        print(f"\nBenchmark Results:")
        print(f"Average time: {avg_time:.2f}s")
        print(f"Min time: {min(times):.2f}s")
        print(f"Max time: {max(times):.2f}s")
        
        return {
            "average": avg_time,
            "min": min(times),
            "max": max(times),
            "all_times": times
        }


def example_style_transfer():
    """Example: Style transfer from black and white to different artistic styles."""
    print("=== Style Transfer Example ===")
    
    pipeline = AdvancedPipeline()
    input_path = "input_images/sample_bw.jpg"
    
    styles = [
        "Renaissance painting",
        "impressionist artwork",
        "vintage 1950s photography",
        "modern cinematic film",
        "watercolor painting",
        "oil painting portrait"
    ]
    
    for i, style in enumerate(styles):
        print(f"\nGenerating style: {style}")
        
        try:
            results = pipeline.generate_with_style_transfer(
                bw_image=input_path,
                style_prompt=style,
                content_weight=0.8
            )
            
            output_path = f"output_images/style_{i+1}_{style.replace(' ', '_')}.png"
            results[0].save(output_path, quality=95)
            print(f"✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Error with style '{style}': {e}")


def example_progressive_refinement():
    """Example: Progressive refinement for highest quality results."""
    print("\n=== Progressive Refinement Example ===")
    
    pipeline = AdvancedPipeline()
    input_path = "input_images/sample_bw.jpg"
    
    base_prompt = "professional portrait, natural skin tones, studio lighting, high quality"
    
    try:
        results = pipeline.generate_progressive_refinement(
            bw_image=input_path,
            base_prompt=base_prompt,
            refinement_steps=3
        )
        
        for i, result in enumerate(results):
            output_path = f"output_images/refinement_step_{i+1}.png"
            result.save(output_path, quality=95)
            print(f"✅ Saved refinement step {i+1}: {output_path}")
        
        print("✅ Progressive refinement complete!")
        
    except Exception as e:
        print(f"❌ Error in progressive refinement: {e}")


def example_parameter_exploration():
    """Example: Explore different parameter combinations."""
    print("\n=== Parameter Exploration Example ===")
    
    pipeline = AdvancedPipeline()
    input_path = "input_images/sample_bw.jpg"
    
    # Parameter combinations to test
    param_combinations = [
        {"guidance_scale": 5.0, "controlnet_scale": 0.8, "steps": 15, "name": "fast_creative"},
        {"guidance_scale": 7.5, "controlnet_scale": 1.0, "steps": 20, "name": "balanced"},
        {"guidance_scale": 10.0, "controlnet_scale": 1.2, "steps": 25, "name": "high_fidelity"},
        {"guidance_scale": 6.0, "controlnet_scale": 0.6, "steps": 20, "name": "loose_interpretation"},
        {"guidance_scale": 9.0, "controlnet_scale": 1.4, "steps": 30, "name": "strict_adherence"}
    ]
    
    base_prompt = "beautiful portrait, natural colors, professional photography"
    
    for params in param_combinations:
        print(f"\nTesting parameters: {params['name']}")
        
        try:
            start_time = time.time()
            
            results = pipeline.generate_colored_image(
                bw_image=input_path,
                prompt=base_prompt,
                guidance_scale=params["guidance_scale"],
                controlnet_conditioning_scale=params["controlnet_scale"],
                num_inference_steps=params["steps"],
                seed=42
            )
            
            generation_time = time.time() - start_time
            
            output_path = f"output_images/params_{params['name']}.png"
            results[0].save(output_path, quality=95)
            
            print(f"✅ Generated in {generation_time:.2f}s: {output_path}")
            
        except Exception as e:
            print(f"❌ Error with parameters {params['name']}: {e}")


def example_batch_with_different_prompts():
    """Example: Process same image with different prompts."""
    print("\n=== Batch Different Prompts Example ===")
    
    pipeline = AdvancedPipeline()
    input_path = "input_images/sample_bw.jpg"
    
    # Different prompt scenarios
    prompts = [
        {
            "name": "natural",
            "prompt": "natural colors, soft lighting, realistic, everyday photography",
            "negative": "oversaturated, artificial, HDR, processed"
        },
        {
            "name": "dramatic",
            "prompt": "dramatic lighting, cinematic colors, moody atmosphere, film noir style",
            "negative": "bright, cheerful, pastel, soft"
        },
        {
            "name": "warm",
            "prompt": "warm golden tones, sunset lighting, cozy atmosphere, golden hour",
            "negative": "cold, blue tones, harsh lighting"
        },
        {
            "name": "cool",
            "prompt": "cool blue tones, modern aesthetic, clean lighting, contemporary",
            "negative": "warm, sepia, vintage, yellow"
        },
        {
            "name": "vintage",
            "prompt": "vintage color palette, retro tones, film photography, nostalgic",
            "negative": "modern, digital, HDR, over-processed"
        }
    ]
    
    for prompt_set in prompts:
        print(f"\nGenerating with '{prompt_set['name']}' style...")
        
        try:
            results = pipeline.generate_colored_image(
                bw_image=input_path,
                prompt=prompt_set["prompt"],
                negative_prompt=prompt_set["negative"],
                seed=42,
                num_inference_steps=20
            )
            
            output_path = f"output_images/prompt_style_{prompt_set['name']}.png"
            results[0].save(output_path, quality=95)
            print(f"✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Error with prompt style '{prompt_set['name']}': {e}")


def example_performance_benchmark():
    """Example: Benchmark performance with different settings."""
    print("\n=== Performance Benchmark Example ===")
    
    pipeline = AdvancedPipeline()
    input_path = "input_images/sample_bw.jpg"
    
    # Test different performance settings
    settings = [
        {"steps": 10, "name": "ultra_fast"},
        {"steps": 15, "name": "fast"},
        {"steps": 20, "name": "balanced"},
        {"steps": 25, "name": "quality"},
        {"steps": 30, "name": "high_quality"}
    ]
    
    base_prompt = "portrait, natural colors, good quality"
    benchmark_results = {}
    
    for setting in settings:
        print(f"\nBenchmarking {setting['name']} ({setting['steps']} steps)...")
        
        try:
            # Run benchmark
            start_time = time.time()
            results = pipeline.generate_colored_image(
                bw_image=input_path,
                prompt=base_prompt,
                num_inference_steps=setting["steps"],
                seed=42
            )
            total_time = time.time() - start_time
            
            # Save result
            output_path = f"output_images/benchmark_{setting['name']}.png"
            results[0].save(output_path, quality=95)
            
            benchmark_results[setting['name']] = {
                "time": total_time,
                "steps": setting["steps"],
                "speed": setting["steps"] / total_time
            }
            
            print(f"✅ {setting['name']}: {total_time:.2f}s ({setting['steps']/total_time:.1f} steps/sec)")
            
        except Exception as e:
            print(f"❌ Error benchmarking {setting['name']}: {e}")
    
    # Save benchmark results
    with open("output_images/benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    print("\n📊 Benchmark Results Summary:")
    for name, data in benchmark_results.items():
        print(f"{name}: {data['time']:.2f}s, {data['speed']:.1f} steps/sec")


if __name__ == "__main__":
    print("🚀 Advanced Stable Diffusion ControlNet Examples")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("output_images", exist_ok=True)
    
    print("\n💡 Available advanced examples:")
    print("1. Style Transfer - Apply different artistic styles")
    print("2. Progressive Refinement - Multi-step quality improvement")
    print("3. Parameter Exploration - Test different generation settings")
    print("4. Different Prompts - Same image, different color interpretations")
    print("5. Performance Benchmark - Speed vs quality analysis")
    
    print("\nUncomment the examples you want to run:")
    
    # Uncomment the examples you want to try:
    
    # example_style_transfer()
    # example_progressive_refinement()
    # example_parameter_exploration()
    # example_batch_with_different_prompts()
    # example_performance_benchmark()
    
    print("\n✨ Place a black and white image at 'input_images/sample_bw.jpg' and uncomment examples to run!")
