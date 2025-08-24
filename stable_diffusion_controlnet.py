#!/usr/bin/env python3
"""
Stable Diffusion ControlNet Pipeline for Black & White to Photorealistic Conversion
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List, Union, Tuple, Dict
import yaml
from pathlib import Path

# Import color palette functionality
try:
    from color_palettes import apply_palette_to_prompt, get_available_palettes, preview_palette
    COLOR_PALETTES_AVAILABLE = True
except ImportError:
    COLOR_PALETTES_AVAILABLE = False
    print("Warning: Color palettes module not available")

from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel,
    DPMSolverMultistepScheduler
)
from controlnet_aux import CannyDetector
import warnings
warnings.filterwarnings("ignore")


class StableDiffusionControlNetPipeline_BWToColor:
    """
    A pipeline for converting black and white images to photorealistic colored images
    using Stable Diffusion with ControlNet conditioning.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['model']['device'] if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16 if self.config['model']['dtype'] == 'float16' and torch.cuda.is_available() else torch.float32
        
        # Initialize components
        self.controlnet = None
        self.pipe = None
        self.canny_detector = CannyDetector()
        
        self._setup_directories()
        self._load_models()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_directories(self):
        """Create necessary directories."""
        for path_key in ['input_dir', 'output_dir', 'cache_dir']:
            path = Path(self.config['paths'][path_key])
            path.mkdir(parents=True, exist_ok=True)
    
    def _load_models(self):
        """Load the ControlNet and Stable Diffusion models."""
        print("Loading ControlNet model...")
        self.controlnet = ControlNetModel.from_pretrained(
            self.config['model']['controlnet_model'],
            torch_dtype=self.dtype,
            cache_dir=self.config['paths']['cache_dir']
        )
        
        print("Loading Stable Diffusion pipeline...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.config['model']['base_model'],
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            cache_dir=self.config['paths']['cache_dir'],
            safety_checker=None,  # Disable safety checker for faster inference
            requires_safety_checker=False
        )
        
        # Optimize pipeline
        self.pipe = self.pipe.to(self.device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        # Enable memory efficient attention if available
        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("Enabled xformers memory efficient attention")
            except Exception as e:
                print(f"Could not enable xformers: {e}")
        
        # Enable model CPU offload to save VRAM
        if self.device.type == 'cuda':
            self.pipe.enable_model_cpu_offload()
            print("Enabled model CPU offload")
    
    def preprocess_bw_image(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[Image.Image, Image.Image]:
        """
        Preprocess black and white image for ControlNet conditioning with enhanced preservation.
        Supports multiple input formats: JPEG, PNG, BMP, TIFF, WEBP, etc.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            Tuple of (original_resized_image, canny_control_image)
        """
        # Load and convert image
        if isinstance(image, str):
            # Enhanced file format support
            supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
            file_ext = Path(image).suffix.lower()
            
            if file_ext not in supported_formats:
                print(f"Warning: File format {file_ext} may not be fully supported. Converting to RGB.")
            
            try:
                image = Image.open(image)
                # Convert to RGB for consistent processing
                if image.mode in ['RGBA', 'LA', 'P']:
                    # Handle transparency and palette modes
                    if image.mode == 'RGBA':
                        # Create white background for transparent images
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                        image = background
                    else:
                        image = image.convert('RGB')
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                    
            except Exception as e:
                print(f"Error loading image {image}: {e}")
                raise
                
        elif isinstance(image, np.ndarray):
            # Handle numpy array input
            if len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA to RGB conversion
                image = Image.fromarray(image).convert('RGB')
            elif len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image)
            elif len(image.shape) == 2:
                # Grayscale to RGB
                image = Image.fromarray(image).convert('RGB')
            else:
                image = Image.fromarray(image)
        
        # Enhanced preprocessing for better preservation
        # 1. Preserve original aspect ratio
        original_size = image.size
        target_size = tuple(self.config['image']['input_size'])
        
        # Calculate aspect ratio preserving resize
        aspect_ratio = original_size[0] / original_size[1]
        target_aspect = target_size[0] / target_size[1]
        
        if aspect_ratio > target_aspect:
            # Image is wider than target
            new_width = target_size[0]
            new_height = int(target_size[0] / aspect_ratio)
        else:
            # Image is taller than target
            new_height = target_size[1]
            new_width = int(target_size[1] * aspect_ratio)
        
        # Resize with high-quality interpolation
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 2. Enhance contrast for better edge detection while preserving details
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        if len(img_array.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced_array = img_array
        
        # Convert back to PIL
        enhanced_image = Image.fromarray(enhanced_array)
        
        # 3. Generate optimized Canny edge detection for better structure preservation
        # Use adaptive thresholds based on image content
        gray = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2GRAY)
        
        # Calculate adaptive thresholds
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Adaptive threshold calculation
        low_threshold = max(50, int(mean_intensity - std_intensity * 0.5))
        high_threshold = min(255, int(mean_intensity + std_intensity * 1.5))
        
        # Ensure minimum difference between thresholds
        if high_threshold - low_threshold < 50:
            high_threshold = min(255, low_threshold + 50)
        
        # Generate Canny edge detection with adaptive thresholds
        canny_image = self.canny_detector(
            enhanced_image,
            low_threshold=low_threshold,
            high_threshold=high_threshold
        )
        
        return enhanced_image, canny_image
    
    def generate_photorealistic_image(
        self,
        bw_image: Union[str, Image.Image, np.ndarray],
        preserve_structure: bool = True,
        enhance_details: bool = True,
        natural_colors: bool = True,
        **generation_kwargs
    ) -> List[Image.Image]:
        """
        Generate photorealistic colored images with maximum preservation of original components.
        
        Args:
            bw_image: Input black and white image
            preserve_structure: Whether to strongly preserve original structure
            enhance_details: Whether to enhance fine details
            natural_colors: Whether to use natural color palette
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated photorealistic images
        """
        # Override generation parameters for photorealistic results
        photorealistic_kwargs = {
            'num_inference_steps': 30,  # More steps for better quality
            'guidance_scale': 9.0,  # Higher guidance for faithfulness
            'controlnet_conditioning_scale': 1.3,  # Stronger control
            'seed': generation_kwargs.get('seed', 42)
        }
        
        # Update with user-provided kwargs
        photorealistic_kwargs.update(generation_kwargs)
        
        # Create specialized prompts for photorealistic results (optimized for token length)
        if preserve_structure:
            structure_prompt = "exact composition, preserve details, maintain features, accurate proportions"
        else:
            structure_prompt = ""
            
        if enhance_details:
            detail_prompt = "high resolution, sharp focus, detailed, realistic texture"
        else:
            detail_prompt = ""
            
        if natural_colors:
            color_prompt = "natural skin tones, authentic colors, realistic lighting"
        else:
            color_prompt = ""
        
        # Combine specialized prompts (concise version)
        specialized_prompt = ", ".join(filter(None, [structure_prompt, detail_prompt, color_prompt]))
        
        # Generate with specialized settings
        return self.generate_colored_image(
            bw_image=bw_image,
            prompt=specialized_prompt,
            **photorealistic_kwargs
        )
    
    def enhance_prompt_for_colorization(self, base_prompt: str = "", color_palette: str = "") -> Tuple[str, str]:
        """
        Create enhanced prompts specifically for photorealistic black and white to color conversion.
        
        Args:
            base_prompt: User-provided base prompt
            color_palette: Name of color palette to apply
            
        Returns:
            Tuple of (positive_prompt, negative_prompt)
        """
        # Enhanced photorealistic prompts that preserve original components
        photorealistic_positive = (
            "photorealistic, exact same composition, identical structure, "
            "natural skin tones, realistic lighting, professional photography, "
            "sharp focus, high resolution, detailed, authentic colors, "
            "preserve original details, maintain facial features, "
            "realistic texture, natural shadows, accurate proportions"
        )
        
        # Strong negative prompts to prevent artistic interpretation
        photorealistic_negative = (
            "artistic, painting, drawing, sketch, cartoon, anime, "
            "digital art, illustration, stylized, abstract, "
            "oversaturated, artificial colors, HDR, over-processed, "
            "blurry, low quality, distorted, deformed, "
            "black and white, monochrome, grayscale, sepia, "
            "fantasy, dreamlike, ethereal, magical"
        )
        
        # Apply color palette if specified
        if color_palette and COLOR_PALETTES_AVAILABLE:
            palette_positive, palette_negative = apply_palette_to_prompt(color_palette, base_prompt)
            if palette_positive:
                photorealistic_positive = f"{palette_positive}, {photorealistic_positive}"
            if palette_negative:
                photorealistic_negative = f"{palette_negative}, {photorealistic_negative}"
        
        # Combine with user prompt and defaults
        if base_prompt:
            positive_prompt = f"{base_prompt}, {photorealistic_positive}, {self.config['prompts']['default_positive']}"
        else:
            positive_prompt = f"{photorealistic_positive}, {self.config['prompts']['default_positive']}"
        
        negative_prompt = f"{photorealistic_negative}, {self.config['prompts']['default_negative']}"
        
        return positive_prompt, negative_prompt
    
    def generate_colored_image(
        self,
        bw_image: Union[str, Image.Image, np.ndarray],
        prompt: str = "",
        negative_prompt: str = "",
        color_palette: str = "",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        controlnet_conditioning_scale: Optional[float] = None,
        seed: Optional[int] = None,
        num_images: int = 1
    ) -> List[Image.Image]:
        """
        Generate colored photorealistic images from black and white input.
        
        Args:
            bw_image: Input black and white image
            prompt: Text prompt describing desired output
            negative_prompt: Negative text prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            controlnet_conditioning_scale: ControlNet conditioning strength
            seed: Random seed for reproducibility
            num_images: Number of images to generate
            
        Returns:
            List of generated colored images
        """
        # Set generation parameters
        num_inference_steps = num_inference_steps or self.config['generation']['num_inference_steps']
        guidance_scale = guidance_scale or self.config['generation']['guidance_scale']
        controlnet_conditioning_scale = controlnet_conditioning_scale or self.config['generation']['controlnet_conditioning_scale']
        seed = seed or self.config['generation']['seed']
        
        # Preprocess input image
        original_image, canny_image = self.preprocess_bw_image(bw_image)
        
        # Enhance prompts for colorization
        if not prompt and not negative_prompt:
            positive_prompt, negative_prompt = self.enhance_prompt_for_colorization(prompt, color_palette)
        else:
            # Apply color palette if specified
            if color_palette and COLOR_PALETTES_AVAILABLE:
                palette_positive, palette_negative = apply_palette_to_prompt(color_palette, prompt)
                positive_prompt = palette_positive + ", " + self.config['prompts']['default_positive'] if palette_positive else prompt + ", " + self.config['prompts']['default_positive'] if prompt else self.config['prompts']['default_positive']
                negative_prompt = (negative_prompt + ", " + palette_negative if negative_prompt else palette_negative) + ", " + self.config['prompts']['default_negative']
            else:
                positive_prompt = prompt + ", " + self.config['prompts']['default_positive'] if prompt else self.config['prompts']['default_positive']
                negative_prompt = negative_prompt + ", " + self.config['prompts']['default_negative'] if negative_prompt else self.config['prompts']['default_negative']
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        print(f"Generating {num_images} colored image(s)...")
        print(f"Positive prompt: {positive_prompt}")
        print(f"Negative prompt: {negative_prompt}")
        
        # Generate images
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            results = self.pipe(
                prompt=[positive_prompt] * num_images,
                negative_prompt=[negative_prompt] * num_images,
                image=canny_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None,
                return_dict=True
            )
        
        return results.images
    
    def process_image_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        prompt: str = "",
        color_palette: str = "",
        **generation_kwargs
    ) -> str:
        """
        Process a single image file and save the result.
        
        Args:
            input_path: Path to input black and white image
            output_path: Path to save output (optional)
            prompt: Text prompt for generation
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Path to the saved output image
        """
        # Generate output path if not provided
        if output_path is None:
            input_name = Path(input_path).stem
            output_path = Path(self.config['paths']['output_dir']) / f"{input_name}_colored.png"
        
        # Generate colored image
        colored_images = self.generate_colored_image(
            bw_image=input_path,
            prompt=prompt,
            color_palette=color_palette,
            **generation_kwargs
        )
        
        # Save the first generated image
        colored_images[0].save(output_path, quality=95)
        print(f"Saved colored image to: {output_path}")
        
        return str(output_path)
    
    def process_photorealistic_image(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        preserve_structure: bool = True,
        enhance_details: bool = True,
        natural_colors: bool = True,
        **generation_kwargs
    ) -> str:
        """
        Process a single image file with photorealistic settings for maximum preservation.
        
        Args:
            input_path: Path to input black and white image
            output_path: Path to save output (optional)
            preserve_structure: Whether to strongly preserve original structure
            enhance_details: Whether to enhance fine details
            natural_colors: Whether to use natural color palette
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Path to the saved photorealistic image
        """
        # Generate output path if not provided
        if output_path is None:
            input_name = Path(input_path).stem
            output_path = Path(self.config['paths']['output_dir']) / f"{input_name}_photorealistic.png"
        
        # Generate photorealistic image
        photorealistic_images = self.generate_photorealistic_image(
            bw_image=input_path,
            preserve_structure=preserve_structure,
            enhance_details=enhance_details,
            natural_colors=natural_colors,
            **generation_kwargs
        )
        
        # Save the first generated image
        photorealistic_images[0].save(output_path, quality=95)
        print(f"Saved photorealistic image to: {output_path}")
        
        return str(output_path)
    
    def process_batch(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        prompt: str = "",
        color_palette: str = "",
        **generation_kwargs
    ) -> List[str]:
        """
        Process a batch of images from a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
            prompt: Text prompt for generation
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of output file paths
        """
        input_dir = Path(input_dir or self.config['paths']['input_dir'])
        output_dir = Path(output_dir or self.config['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Find all image files
        image_files = [
            f for f in input_dir.glob('*')
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return []
        
        print(f"Found {len(image_files)} images to process")
        
        output_paths = []
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            output_path = output_dir / f"{image_file.stem}_colored.png"
            result_path = self.process_image_file(
                str(image_file),
                str(output_path),
                prompt,
                color_palette,
                **generation_kwargs
            )
            output_paths.append(result_path)
        
        print(f"Batch processing complete! Processed {len(output_paths)} images.")
        return output_paths
    
    def list_available_palettes(self) -> Dict[str, List[str]]:
        """List all available color palettes."""
        if COLOR_PALETTES_AVAILABLE:
            return get_available_palettes()
        return {"predefined": [], "custom": []}
    
    def get_palette_info(self, palette_name: str) -> Dict:
        """Get information about a specific color palette."""
        if COLOR_PALETTES_AVAILABLE:
            return preview_palette(palette_name)
        return {}


def main():
    """Example usage of the pipeline."""
    # Initialize the pipeline
    pipeline = StableDiffusionControlNetPipeline_BWToColor()
    
    # Example: Process a single image
    # pipeline.process_image_file(
    #     input_path="path/to/your/bw_image.jpg",
    #     prompt="beautiful portrait, natural lighting, warm colors"
    # )
    
    # Example: Process batch of images
    # pipeline.process_batch(
    #     prompt="photorealistic, vibrant colors, professional photography"
    # )
    
    print("Pipeline initialized successfully!")
    print("Use pipeline.process_image_file() for single images or pipeline.process_batch() for batch processing")


if __name__ == "__main__":
    main()
