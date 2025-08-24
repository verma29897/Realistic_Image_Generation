# 🎨 Stable Diffusion ControlNet: Black & White to Photorealistic

Transform your black and white images into stunning photorealistic colored images using the power of Stable Diffusion with ControlNet conditioning.

![Project Banner](https://img.shields.io/badge/AI-Stable%20Diffusion-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- **🖼️ Black & White to Color**: Transform B&W images to photorealistic colored versions
- **🎯 ControlNet Precision**: Maintain original image structure and composition
- **📸 Photorealistic Mode**: Maximum preservation of original photo components
- **🎨 Multiple Styles**: Support for various artistic styles and color palettes
- **🖼️ Image Palette Extraction**: Extract color palettes from any image
- **📁 Multi-Format Support**: JPEG, PNG, BMP, TIFF, WEBP, GIF, and more
- **🚀 Easy to Use**: Simple API and beautiful web interface
- **⚡ Optimized Performance**: GPU acceleration with memory optimization
- **🔧 Highly Configurable**: Extensive customization options
- **📱 Web Interface**: User-friendly Gradio interface
- **🎪 Batch Processing**: Process multiple images at once

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- At least 8GB RAM (16GB+ recommended for GPU)
- 10GB+ free disk space for models

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ai_code
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure settings (optional):**
   ```bash
   # Edit config.yaml to customize model settings
   nano config.yaml
   ```

4. **Create input/output directories:**
   ```bash
   mkdir -p input_images output_images
   ```

### GPU Setup (Recommended)

For optimal performance, ensure you have:
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- cuDNN libraries

Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## 🚀 Quick Start

### Method 1: Web Interface (Easiest)

Launch the beautiful web interface:

```bash
python gradio_interface.py
```

Then open your browser to `http://localhost:7860` and start converting images!

### Method 2: Python API

```python
from stable_diffusion_controlnet import StableDiffusionControlNetPipeline_BWToColor

# Initialize the pipeline
pipeline = StableDiffusionControlNetPipeline_BWToColor()

# Convert a single image (standard mode)
output_path = pipeline.process_image_file(
    input_path="input_images/my_bw_photo.jpg",
    prompt="beautiful portrait, natural skin tones, warm lighting"
)

# Convert with photorealistic settings (maximum preservation)
photorealistic_path = pipeline.process_photorealistic_image(
    input_path="input_images/my_bw_photo.jpg",
    preserve_structure=True,
    enhance_details=True,
    natural_colors=True
)

print(f"Colored image saved to: {output_path}")
print(f"Photorealistic image saved to: {photorealistic_path}")
```

### Method 3: Command Line

```bash
# Basic usage
python run.py --single input_images/photo.jpg --prompt "vibrant colors, natural lighting"

# Photorealistic mode (maximum preservation)
python run.py --single input_images/photo.jpg --photorealistic

# With color palette
python run.py --single input_images/photo.jpg --palette natural_warm

# Extract palette from image
python run.py --extract-palette reference_image.jpg --palette-name "my_palette" --num-colors 5

# Validate image format
python run.py --validate-image your_image.png
python run.py --validate-image your_image.tiff

# Get image information
python run.py --image-info your_image.jpg

# List supported formats
python run.py --supported-formats

# List available color palettes
python run.py --list-palettes
```

## 📖 Usage Examples

### Single Image Processing

```python
from stable_diffusion_controlnet import StableDiffusionControlNetPipeline_BWToColor

# Initialize pipeline
pipeline = StableDiffusionControlNetPipeline_BWToColor()

# Process with custom settings
output_path = pipeline.process_image_file(
    input_path="input_images/portrait.jpg",
    prompt="professional headshot, natural skin tones, studio lighting",
    negative_prompt="blurry, low quality, oversaturated",
    num_inference_steps=25,
    guidance_scale=7.5,
    seed=42
)
```

### Image Palette Extraction

```python
from color_palettes import extract_palette_from_image, add_palette_from_image

# Extract palette from image
palette_data = extract_palette_from_image(
    image_path="reference_image.jpg",
    num_colors=5,
    method="kmeans"  # or "dominant"
)

# Save as custom palette
add_palette_from_image(
    image_path="reference_image.jpg",
    palette_name="my_custom_palette",
    num_colors=5,
    method="kmeans"
)

# Use extracted palette for colorization
output_path = pipeline.process_image_file(
    input_path="input_images/bw_photo.jpg",
    color_palette="my_custom_palette"
)
```

### Multi-Format Image Support

```python
from image_utils import ImageFormatHandler, validate_image_file

# Validate any image format
is_valid, message = validate_image_file("your_image.png")
is_valid, message = validate_image_file("your_image.tiff")
is_valid, message = validate_image_file("your_image.webp")

# Get detailed image information
info = ImageFormatHandler.get_image_info("your_image.jpg")
print(f"Format: {info['format']}")
print(f"Size: {info['width']}x{info['height']}")
print(f"Mode: {info['mode']}")

# List supported formats
formats = ImageFormatHandler.get_supported_formats_list()
print(f"Supported: {formats}")

# Process any supported format
pipeline.process_photorealistic_image("input.png")  # PNG
pipeline.process_photorealistic_image("input.tiff") # TIFF
pipeline.process_photorealistic_image("input.webp") # WEBP
```

### Batch Processing

```python
# Process all images in input directory
output_paths = pipeline.process_batch(
    prompt="photorealistic, vibrant colors, professional photography",
    num_inference_steps=20
)

print(f"Processed {len(output_paths)} images successfully!")
```

### Advanced Generation

```python
# Generate multiple variations
colored_images = pipeline.generate_colored_image(
    bw_image="input_images/photo.jpg",
    prompt="warm colors, golden hour lighting, cinematic",
    num_images=4,  # Generate 4 variations
    seed=42
)

# Save each variation
for i, img in enumerate(colored_images):
    img.save(f"output_images/variation_{i+1}.png")
```

## 🎯 Prompt Engineering Tips

### For Best Results

**Positive Prompts:**
- "photorealistic, high quality, detailed"
- "natural skin tones, warm lighting"
- "vibrant colors, professional photography"
- "cinematic lighting, sharp focus"

**Negative Prompts:**
- "blurry, low quality, distorted"
- "oversaturated, artificial colors"
- "cartoon, anime, painting"
- "black and white, monochrome"

### Example Prompt Combinations

| Image Type | Positive Prompt | Negative Prompt |
|------------|----------------|-----------------|
| **Portraits** | "natural skin tones, soft lighting, professional headshot, warm colors" | "blurry, oversaturated, artificial, cartoon" |
| **Landscapes** | "vibrant nature colors, golden hour, cinematic landscape, detailed" | "dull, monochrome, low contrast, artificial" |
| **Architecture** | "realistic colors, natural lighting, architectural photography" | "oversaturated, HDR, processed, artificial" |
| **Vintage Photos** | "restored vintage colors, film photography, nostalgic tones" | "modern, digital, oversaturated, artificial" |

## ⚙️ Configuration

### Main Configuration (`config.yaml`)

```yaml
model:
  base_model: "runwayml/stable-diffusion-v1-5"
  controlnet_model: "lllyasviel/sd-controlnet-canny"
  device: "cuda"  # or "cpu"
  dtype: "float16"  # or "float32"

generation:
  num_inference_steps: 20
  guidance_scale: 7.5
  controlnet_conditioning_scale: 1.0
  seed: 42

image:
  input_size: [512, 512]
  canny_low_threshold: 100
  canny_high_threshold: 200
```

### Performance Tuning

| Setting | Fast | Balanced | Quality |
|---------|------|----------|---------|
| **Steps** | 10-15 | 20-25 | 30-50 |
| **Guidance Scale** | 5-7 | 7-8 | 8-12 |
| **Resolution** | 512x512 | 512x512 | 768x768 |
| **ControlNet Scale** | 0.8-1.0 | 1.0-1.2 | 1.2-1.5 |

## 📁 Project Structure

```
ai_code/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Main configuration
├── stable_diffusion_controlnet.py     # Core pipeline
├── gradio_interface.py                # Web interface
├── examples/
│   ├── basic_usage.py                 # Basic examples
│   └── advanced_usage.py              # Advanced examples
├── input_images/                      # Place your B&W images here
├── output_images/                     # Generated colored images
└── model_cache/                       # Downloaded models cache
```

## 🎪 Advanced Features

### Style Transfer

```python
# Apply artistic styles
pipeline = AdvancedPipeline()
results = pipeline.generate_with_style_transfer(
    bw_image="input.jpg",
    style_prompt="Renaissance painting",
    content_weight=0.8
)
```

### Progressive Refinement

```python
# Multi-step quality improvement
results = pipeline.generate_progressive_refinement(
    bw_image="input.jpg",
    base_prompt="professional portrait",
    refinement_steps=3
)
```

### Parameter Exploration

Test different settings automatically:

```python
# Run parameter exploration
python examples/advanced_usage.py
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Enable CPU offload or reduce batch size
pipeline.pipe.enable_model_cpu_offload()
```

**2. Slow Generation**
```python
# Solution: Reduce inference steps or use float16
num_inference_steps=15  # Instead of 25
dtype="float16"  # Instead of "float32"
```

**3. Poor Quality Results**
- Use higher contrast B&W images
- Increase inference steps (20-30)
- Adjust ControlNet conditioning scale (0.8-1.5)
- Improve prompts with more descriptive terms

**4. Model Download Issues**
```bash
# Clear cache and retry
rm -rf model_cache/
python stable_diffusion_controlnet.py
```

### Performance Optimization

**For GPU:**
- Use `torch.compile()` for faster inference (PyTorch 2.0+)
- Enable xformers memory efficient attention
- Use float16 precision

**For CPU:**
- Reduce image resolution to 256x256 or 384x384
- Use fewer inference steps (10-15)
- Process images one at a time

## 📊 Benchmarks

| Hardware | Resolution | Steps | Time | Quality |
|----------|------------|-------|------|---------|
| RTX 4090 | 512x512 | 20 | ~3s | Excellent |
| RTX 3080 | 512x512 | 20 | ~5s | Excellent |
| RTX 2060 | 512x512 | 15 | ~8s | Good |
| CPU (16 cores) | 384x384 | 10 | ~60s | Fair |

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run tests
pytest tests/

# Format code
black .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stable Diffusion** by Stability AI
- **ControlNet** by Zhang et al.
- **Diffusers** library by Hugging Face
- **Gradio** for the web interface

## 🔗 Links

- [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release)
- [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Gradio Documentation](https://gradio.app/docs/)

## 📞 Support

- Create an issue for bug reports
- Join our discussions for questions
- Check the troubleshooting section above

---

<div align="center">

**🌟 If you find this project helpful, please give it a star! 🌟**

Made with ❤️ using Stable Diffusion + ControlNet

</div>
