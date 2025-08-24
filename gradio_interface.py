#!/usr/bin/env python3
"""
Gradio Web Interface for Stable Diffusion ControlNet Black & White to Color Conversion
"""

import gradio as gr
import os
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import tempfile
import time

from stable_diffusion_controlnet import StableDiffusionControlNetPipeline_BWToColor

# Import color palette functionality
try:
    from color_palettes import get_available_palettes, preview_palette, extract_palette_from_image, add_palette_from_image
    COLOR_PALETTES_AVAILABLE = True
except ImportError:
    COLOR_PALETTES_AVAILABLE = False


class GradioInterface:
    """Gradio web interface for the black and white to color conversion pipeline."""
    
    def __init__(self):
        self.pipeline = None
        self.is_initialized = False
    
    def initialize_pipeline(self):
        """Initialize the pipeline (lazy loading)."""
        if not self.is_initialized:
            try:
                print("Initializing Stable Diffusion ControlNet pipeline...")
                self.pipeline = StableDiffusionControlNetPipeline_BWToColor()
                self.is_initialized = True
                return "✅ Pipeline initialized successfully!"
            except Exception as e:
                error_msg = f"❌ Error initializing pipeline: {str(e)}"
                print(error_msg)
                return error_msg
        return "✅ Pipeline already initialized!"
    
    def process_single_image(
        self,
        input_image,
        prompt,
        negative_prompt,
        color_palette,
        num_inference_steps,
        guidance_scale,
        controlnet_conditioning_scale,
        seed,
        num_images
    ):
        """Process a single uploaded image."""
        if not self.is_initialized:
            init_msg = self.initialize_pipeline()
            if "Error" in init_msg:
                return None, init_msg
        
        if input_image is None:
            return None, "❌ Please upload an image first."
        
        try:
            start_time = time.time()
            
            # Convert gradio image to PIL if needed
            if isinstance(input_image, np.ndarray):
                input_image = Image.fromarray(input_image)
            
            # Generate colored images
            colored_images = self.pipeline.generate_colored_image(
                bw_image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                color_palette=color_palette,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                seed=int(seed) if seed >= 0 else None,
                num_images=int(num_images)
            )
            
            processing_time = time.time() - start_time
            
            # Return the first image and info
            result_image = colored_images[0]
            info_msg = f"✅ Generated {len(colored_images)} image(s) in {processing_time:.2f} seconds"
            
            return result_image, info_msg
            
        except Exception as e:
            error_msg = f"❌ Error processing image: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def create_interface(self):
        """Create and return the Gradio interface."""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .settings-panel {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        """
        
        with gr.Blocks(css=css, title="Black & White to Photorealistic Converter", theme=gr.themes.Soft()) as interface:
            
            # Header
            gr.HTML("""
                <div class="main-header">
                    <h1>🎨 Black & White to Photorealistic Converter</h1>
                    <p>Transform your black and white images into stunning photorealistic colored images using Stable Diffusion + ControlNet</p>
                </div>
            """)
            
            # Initialize pipeline button
            with gr.Row():
                init_btn = gr.Button("🚀 Initialize Pipeline", variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False, value="Click 'Initialize Pipeline' to start")
            
            init_btn.click(
                fn=self.initialize_pipeline,
                outputs=status_text
            )
            
            # Main interface
            with gr.Row():
                # Input column
                with gr.Column(scale=1):
                    gr.HTML('<h3>📤 Input</h3>')
                    
                    input_image = gr.Image(
                        label="Upload Black & White Image (JPEG, PNG, BMP, TIFF, WEBP, GIF)",
                        type="pil",
                        height=400,
                        file_types=["image"]
                    )
                    
                    prompt = gr.Textbox(
                        label="Prompt (Optional)",
                        placeholder="e.g., 'beautiful portrait, natural lighting, warm colors'",
                        lines=3,
                        value=""
                    )
                    
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt (Optional)",
                        placeholder="e.g., 'blurry, low quality, distorted'",
                        lines=2,
                        value=""
                    )
                    
                    # Color palette selection
                    if COLOR_PALETTES_AVAILABLE:
                        color_palette = gr.Dropdown(
                            label="🎨 Color Palette (Optional)",
                            choices=["None"] + list(get_available_palettes()["predefined"]),
                            value="None",
                            info="Choose a color palette to apply specific color styles"
                        )
                        
                        # Palette preview
                        palette_preview = gr.HTML(
                            value="<div style='text-align: center; color: #666;'>Select a palette to see preview</div>",
                            label="Palette Preview"
                        )
                        
                        def update_palette_preview(palette_name):
                            if palette_name and palette_name != "None":
                                try:
                                    palette_info = preview_palette(palette_name)
                                    if palette_info:
                                        colors_html = ""
                                        for color in palette_info.get("colors", []):
                                            colors_html += f"<div style='display: inline-block; width: 30px; height: 30px; background-color: {color}; border: 1px solid #ccc; margin: 2px; border-radius: 4px;' title='{color}'></div>"
                                        
                                        preview_html = f"""
                                        <div style='padding: 10px; border: 1px solid #ddd; border-radius: 8px; background: #f9f9f9;'>
                                            <h4 style='margin: 0 0 8px 0;'>{palette_info['name']}</h4>
                                            <p style='margin: 0 0 8px 0; font-size: 12px; color: #666;'>{palette_info['description']}</p>
                                            <div style='margin: 8px 0;'>{colors_html}</div>
                                            <p style='margin: 4px 0; font-size: 11px; color: #888;'><strong>Style:</strong> {palette_info.get('prompt_addition', 'N/A')}</p>
                                        </div>
                                        """
                                        return preview_html
                                except Exception as e:
                                    return f"<div style='color: red;'>Error loading palette: {str(e)}</div>"
                            return "<div style='text-align: center; color: #666;'>Select a palette to see preview</div>"
                        
                        # Update palette preview when selection changes
                        color_palette.change(
                            fn=update_palette_preview,
                            inputs=[color_palette],
                            outputs=[palette_preview]
                        )
                        
                        # Image palette extraction section
                        with gr.Accordion("🖼️ Extract Palette from Image", open=False):
                            gr.HTML("""
                                <p>Upload an image to extract its color palette and use it for colorization.</p>
                            """)
                            
                            palette_image_input = gr.Image(
                                label="Upload Image for Palette Extraction (JPEG, PNG, BMP, TIFF, WEBP, GIF)",
                                type="filepath",
                                file_types=["image"]
                            )
                            
                            with gr.Row():
                                palette_name_input = gr.Textbox(
                                    label="Palette Name (Optional)",
                                    placeholder="e.g., 'my_custom_palette'",
                                    value=""
                                )
                                num_colors_input = gr.Slider(
                                    label="Number of Colors",
                                    minimum=3,
                                    maximum=10,
                                    value=5,
                                    step=1
                                )
                            
                            extraction_method = gr.Dropdown(
                                label="Extraction Method",
                                choices=["kmeans", "dominant"],
                                value="kmeans",
                                info="K-means: Clusters similar colors. Dominant: Finds most common colors."
                            )
                            
                            extract_btn = gr.Button("🎨 Extract Palette", variant="secondary")
                            extraction_result = gr.Textbox(
                                label="Extraction Result",
                                interactive=False
                            )
                            
                            def extract_and_add_palette(image_path, palette_name, num_colors, method):
                                if not image_path:
                                    return "❌ Please upload an image first."
                                
                                try:
                                    # Extract palette
                                    palette_data = extract_palette_from_image(image_path, num_colors, method)
                                    
                                    if not palette_data:
                                        return "❌ Failed to extract palette from image."
                                    
                                    # Add to custom palettes
                                    success = add_palette_from_image(
                                        image_path, 
                                        palette_name if palette_name else None,
                                        num_colors, 
                                        method
                                    )
                                    
                                    if success:
                                        palette_name_used = palette_name if palette_name else palette_data["name"]
                                        return f"✅ Palette '{palette_name_used}' extracted and added successfully!\nColors: {', '.join(palette_data['colors'])}"
                                    else:
                                        return "❌ Failed to add palette to custom palettes."
                                        
                                except Exception as e:
                                    return f"❌ Error: {str(e)}"
                            
                            extract_btn.click(
                                fn=extract_and_add_palette,
                                inputs=[palette_image_input, palette_name_input, num_colors_input, extraction_method],
                                outputs=[extraction_result]
                            )
                    else:
                        color_palette = gr.Dropdown(
                            label="🎨 Color Palette (Optional)",
                            choices=["None"],
                            value="None",
                            info="Color palettes not available"
                        )
                
                # Output column
                with gr.Column(scale=1):
                    gr.HTML('<h3>📤 Output</h3>')
                    
                    output_image = gr.Image(
                        label="Generated Colored Image",
                        height=400
                    )
                    
                    info_text = gr.Textbox(
                        label="Generation Info",
                        interactive=False,
                        lines=2
                    )
            
            # Advanced settings
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=1,
                        info="More steps = better quality but slower"
                    )
                    
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        info="How closely to follow the prompt"
                    )
                
                with gr.Row():
                    controlnet_conditioning_scale = gr.Slider(
                        label="ControlNet Strength",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        info="How strongly to follow the image structure"
                    )
                    
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0,
                        info="Use same seed for reproducible results"
                    )
                
                num_images = gr.Slider(
                    label="Number of Images",
                    minimum=1,
                    maximum=4,
                    value=1,
                    step=1,
                    info="Number of variations to generate"
                )
            
            # Generate button
            generate_btn = gr.Button("🎨 Generate Colored Image", variant="primary", size="lg")
            
            # Connect the generate function
            generate_btn.click(
                fn=self.process_single_image,
                inputs=[
                    input_image,
                    prompt,
                    negative_prompt,
                    color_palette,
                    num_inference_steps,
                    guidance_scale,
                    controlnet_conditioning_scale,
                    seed,
                    num_images
                ],
                outputs=[output_image, info_text]
            )
            
            # Example images and tips
            with gr.Accordion("💡 Tips & Examples", open=False):
                gr.HTML("""
                    <h4>Tips for better results:</h4>
                    <ul>
                        <li><strong>High contrast images work best:</strong> Images with clear edges and good contrast will produce better results</li>
                        <li><strong>Describe the desired outcome:</strong> Use prompts like "warm skin tones", "natural lighting", "vibrant colors"</li>
                        <li><strong>Use negative prompts:</strong> Specify what you don't want, like "oversaturated", "artificial colors"</li>
                        <li><strong>Experiment with ControlNet strength:</strong> Lower values give more creative freedom, higher values follow the structure more closely</li>
                        <li><strong>Adjust inference steps:</strong> 15-20 steps are usually sufficient for good quality</li>
                    </ul>
                    
                    <h4>Example prompts:</h4>
                    <ul>
                        <li><strong>Portraits:</strong> "natural skin tones, soft lighting, professional photography, warm colors"</li>
                        <li><strong>Landscapes:</strong> "vibrant nature colors, golden hour lighting, cinematic, detailed"</li>
                        <li><strong>Architecture:</strong> "realistic colors, natural lighting, architectural photography, high detail"</li>
                        <li><strong>Vintage photos:</strong> "restored vintage colors, film photography, nostalgic tones"</li>
                    </ul>
                """)
            
            # Footer
            gr.HTML("""
                <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #eee;">
                    <p>Built with Stable Diffusion + ControlNet | Powered by 🤗 Diffusers & Gradio</p>
                </div>
            """)
        
        return interface


def main():
    """Launch the Gradio interface."""
    # Check if CUDA is available
    device_info = "🔥 CUDA GPU" if torch.cuda.is_available() else "💻 CPU"
    print(f"Running on: {device_info}")
    
    # Create and launch interface
    app = GradioInterface()
    interface = app.create_interface()
    
    # Launch with public sharing disabled by default for security
    interface.launch(
        server_name="0.0.0.0",  # Allow access from other devices on network
        server_port=7860,
        share=False,  # Set to True if you want a public gradio.live link
        show_api=False,
        inbrowser=True,  # Automatically open in browser
        show_error=True
    )


if __name__ == "__main__":
    main()
