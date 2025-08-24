#!/usr/bin/env python3
"""
Color Palette Management for Stable Diffusion ControlNet
Provides predefined color palettes and custom color options for image colorization
"""

from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import colorsys
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import colorsys


class ColorPaletteManager:
    """Manages color palettes for image colorization."""
    
    def __init__(self):
        self.palettes = self._load_predefined_palettes()
        self.custom_palettes = {}
        self.palette_file = Path("custom_palettes.json")
        self._load_custom_palettes()
    
    def _load_predefined_palettes(self) -> Dict[str, Dict]:
        """Load predefined color palettes."""
        return {
            # Natural & Realistic
            "natural_warm": {
                "name": "Natural Warm",
                "description": "Warm, natural skin tones and golden lighting",
                "colors": ["#F4D03F", "#E67E22", "#D35400", "#A040A3", "#8E44AD"],
                "prompt_addition": "warm golden tones, natural skin tones, golden hour lighting, warm colors",
                "negative_addition": "cool tones, blue lighting, cold colors"
            },
            "natural_cool": {
                "name": "Natural Cool", 
                "description": "Cool, natural tones with blue undertones",
                "colors": ["#3498DB", "#2980B9", "#5DADE2", "#85C1E9", "#AED6F1"],
                "prompt_addition": "cool blue tones, natural lighting, soft blue undertones, cool colors",
                "negative_addition": "warm tones, orange lighting, warm colors"
            },
            "natural_neutral": {
                "name": "Natural Neutral",
                "description": "Balanced, neutral colors for realistic appearance",
                "colors": ["#95A5A6", "#7F8C8D", "#BDC3C7", "#ECF0F1", "#F8F9F9"],
                "prompt_addition": "neutral colors, balanced tones, natural lighting, realistic colors",
                "negative_addition": "oversaturated, artificial colors, extreme tones"
            },
            
            # Artistic Styles
            "vintage_sepia": {
                "name": "Vintage Sepia",
                "description": "Classic sepia tones for vintage look",
                "colors": ["#8B4513", "#A0522D", "#CD853F", "#DEB887", "#F5DEB3"],
                "prompt_addition": "sepia tones, vintage colors, nostalgic, film photography, warm browns",
                "negative_addition": "modern colors, bright colors, digital look"
            },
            "vintage_kodachrome": {
                "name": "Vintage Kodachrome",
                "description": "Classic Kodachrome film colors",
                "colors": ["#FF6B35", "#F7931E", "#FFD23F", "#06FFA5", "#1B9AAA"],
                "prompt_addition": "kodachrome film colors, vintage saturated colors, classic photography",
                "negative_addition": "modern digital, muted colors, contemporary"
            },
            "cinematic": {
                "name": "Cinematic",
                "description": "Movie-like color grading with dramatic tones",
                "colors": ["#2C3E50", "#34495E", "#E74C3C", "#C0392B", "#8E44AD"],
                "prompt_addition": "cinematic lighting, dramatic colors, film noir style, moody atmosphere",
                "negative_addition": "bright cheerful colors, pastel tones, soft lighting"
            },
            
            # Modern Styles
            "modern_minimal": {
                "name": "Modern Minimal",
                "description": "Clean, minimal color palette",
                "colors": ["#FFFFFF", "#F8F9FA", "#E9ECEF", "#6C757D", "#495057"],
                "prompt_addition": "minimal colors, clean aesthetic, modern design, subtle tones",
                "negative_addition": "bright colors, saturated, busy patterns"
            },
            "modern_vibrant": {
                "name": "Modern Vibrant",
                "description": "Bright, modern vibrant colors",
                "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
                "prompt_addition": "vibrant modern colors, bright contemporary, saturated colors",
                "negative_addition": "muted colors, vintage, soft tones"
            },
            
            # Seasonal
            "autumn": {
                "name": "Autumn",
                "description": "Warm autumn colors with orange and brown tones",
                "colors": ["#D2691E", "#CD853F", "#8B4513", "#A0522D", "#F4A460"],
                "prompt_addition": "autumn colors, warm fall tones, golden browns, seasonal lighting",
                "negative_addition": "cool colors, spring tones, bright greens"
            },
            "winter": {
                "name": "Winter",
                "description": "Cool winter colors with blue and white tones",
                "colors": ["#87CEEB", "#B0E0E6", "#F0F8FF", "#4682B4", "#191970"],
                "prompt_addition": "winter colors, cool blue tones, crisp lighting, seasonal atmosphere",
                "negative_addition": "warm colors, summer tones, bright yellows"
            },
            "spring": {
                "name": "Spring",
                "description": "Fresh spring colors with green and pastel tones",
                "colors": ["#90EE90", "#98FB98", "#00FF7F", "#32CD32", "#228B22"],
                "prompt_addition": "spring colors, fresh green tones, natural lighting, seasonal vibrancy",
                "negative_addition": "autumn colors, warm browns, muted tones"
            },
            "summer": {
                "name": "Summer",
                "description": "Bright summer colors with warm and vibrant tones",
                "colors": ["#FFD700", "#FFA500", "#FF6347", "#FF69B4", "#00CED1"],
                "prompt_addition": "summer colors, bright warm tones, sunny lighting, vibrant atmosphere",
                "negative_addition": "cool colors, winter tones, muted lighting"
            },
            
            # Professional
            "professional_portrait": {
                "name": "Professional Portrait",
                "description": "Professional portrait photography colors",
                "colors": ["#F5F5DC", "#DEB887", "#CD853F", "#8B4513", "#654321"],
                "prompt_addition": "professional portrait colors, studio lighting, natural skin tones, elegant",
                "negative_addition": "casual colors, bright lighting, amateur look"
            },
            "fashion": {
                "name": "Fashion",
                "description": "High-fashion photography color palette",
                "colors": ["#000000", "#FFFFFF", "#C0C0C0", "#FFD700", "#FF1493"],
                "prompt_addition": "fashion photography, high-end colors, sophisticated lighting, elegant",
                "negative_addition": "casual colors, everyday lighting, simple"
            },
            
            # Creative
            "fantasy": {
                "name": "Fantasy",
                "description": "Fantasy and magical color palette",
                "colors": ["#9370DB", "#8A2BE2", "#FF1493", "#00CED1", "#FFD700"],
                "prompt_addition": "fantasy colors, magical lighting, ethereal tones, dreamlike",
                "negative_addition": "realistic colors, natural lighting, mundane"
            },
            "cyberpunk": {
                "name": "Cyberpunk",
                "description": "Cyberpunk neon and dark color palette",
                "colors": ["#00FFFF", "#FF00FF", "#00FF00", "#000000", "#1E1E1E"],
                "prompt_addition": "cyberpunk colors, neon lighting, futuristic, digital atmosphere",
                "negative_addition": "natural colors, organic lighting, vintage"
            }
        }
    
    def _load_custom_palettes(self):
        """Load custom palettes from file."""
        if self.palette_file.exists():
            try:
                with open(self.palette_file, 'r') as f:
                    self.custom_palettes = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load custom palettes: {e}")
                self.custom_palettes = {}
    
    def _save_custom_palettes(self):
        """Save custom palettes to file."""
        try:
            with open(self.palette_file, 'w') as f:
                json.dump(self.custom_palettes, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save custom palettes: {e}")
    
    def get_palette(self, palette_name: str) -> Optional[Dict]:
        """Get a palette by name."""
        if palette_name in self.palettes:
            return self.palettes[palette_name]
        elif palette_name in self.custom_palettes:
            return self.custom_palettes[palette_name]
        return None
    
    def list_palettes(self) -> Dict[str, List[str]]:
        """List all available palettes."""
        return {
            "predefined": list(self.palettes.keys()),
            "custom": list(self.custom_palettes.keys())
        }
    
    def add_custom_palette(self, name: str, colors: List[str], description: str = "", 
                          prompt_addition: str = "", negative_addition: str = ""):
        """Add a custom color palette."""
        self.custom_palettes[name] = {
            "name": name,
            "description": description,
            "colors": colors,
            "prompt_addition": prompt_addition,
            "negative_addition": negative_addition
        }
        self._save_custom_palettes()
    
    def extract_palette_from_image(self, image_path: str, num_colors: int = 5, method: str = "kmeans") -> Dict:
        """
        Extract color palette from an image.
        Supports multiple formats: JPEG, PNG, BMP, TIFF, WEBP, GIF, etc.
        
        Args:
            image_path: Path to the image file
            num_colors: Number of colors to extract
            method: Extraction method ('kmeans', 'dominant', 'median_cut')
            
        Returns:
            Dictionary containing palette information
        """
        try:
            # Enhanced image loading with format support
            supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
            file_ext = Path(image_path).suffix.lower()
            
            if file_ext not in supported_formats:
                print(f"Warning: File format {file_ext} may not be fully supported. Attempting to load anyway.")
            
            # Load image with enhanced format handling
            image = Image.open(image_path)
            
            # Handle different image modes
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
            
            # Resize for faster processing
            image_small = image.resize((150, 150), Image.Resampling.LANCZOS)
            img_array = np.array(image_small)
            
            if method == "kmeans":
                colors = self._extract_kmeans_palette(img_array, num_colors)
            elif method == "dominant":
                colors = self._extract_dominant_colors(img_array, num_colors)
            else:
                colors = self._extract_kmeans_palette(img_array, num_colors)
            
            # Convert to hex format
            hex_colors = [self._rgb_to_hex(color) for color in colors]
            
            # Generate palette name from image filename
            palette_name = Path(image_path).stem + "_extracted"
            
            return {
                "name": palette_name,
                "description": f"Extracted from {Path(image_path).name} using {method} method",
                "colors": hex_colors,
                "source_image": image_path,
                "extraction_method": method,
                "num_colors": num_colors
            }
            
        except Exception as e:
            print(f"Error extracting palette from image: {e}")
            return {}
    
    def _extract_kmeans_palette(self, img_array: np.ndarray, num_colors: int) -> List[Tuple[int, int, int]]:
        """Extract palette using K-means clustering."""
        # Reshape image to 2D array of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Sort by frequency (most common first)
        labels = kmeans.labels_
        color_counts = np.bincount(labels)
        sorted_indices = np.argsort(color_counts)[::-1]
        colors = colors[sorted_indices]
        
        return [tuple(color) for color in colors]
    
    def _extract_dominant_colors(self, img_array: np.ndarray, num_colors: int) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using histogram analysis."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Create histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Find peaks in histograms
        colors = []
        for i in range(num_colors):
            # Find dominant hue
            h_peak = np.argmax(h_hist)
            h_hist[h_peak] = 0  # Remove peak for next iteration
            
            # Find corresponding saturation and value
            s_peak = np.argmax(s_hist)
            v_peak = np.argmax(v_hist)
            
            # Convert back to RGB
            hsv_color = np.array([[[h_peak, s_peak, v_peak]]], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
            
            colors.append(tuple(rgb_color))
        
        return colors
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color string."""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def add_palette_from_image(self, image_path: str, palette_name: str = None, 
                              num_colors: int = 5, method: str = "kmeans") -> bool:
        """
        Extract palette from image and add it to custom palettes.
        
        Args:
            image_path: Path to the image file
            palette_name: Custom name for the palette (optional)
            num_colors: Number of colors to extract
            method: Extraction method
            
        Returns:
            True if successful, False otherwise
        """
        palette_data = self.extract_palette_from_image(image_path, num_colors, method)
        
        if not palette_data:
            return False
        
        # Use custom name if provided
        if palette_name:
            palette_data["name"] = palette_name
        
        # Add to custom palettes
        self.custom_palettes[palette_data["name"]] = palette_data
        self._save_custom_palettes()
        
        return True
    
    def remove_custom_palette(self, name: str) -> bool:
        """Remove a custom palette."""
        if name in self.custom_palettes:
            del self.custom_palettes[name]
            self._save_custom_palettes()
            return True
        return False
    
    def get_palette_prompt(self, palette_name: str, base_prompt: str = "") -> Tuple[str, str]:
        """Get enhanced prompts for a specific palette."""
        palette = self.get_palette(palette_name)
        if not palette:
            return base_prompt, ""
        
        positive = base_prompt
        if palette.get("prompt_addition"):
            positive += f", {palette['prompt_addition']}"
        
        negative = palette.get("negative_addition", "")
        return positive, negative
    
    def preview_palette(self, palette_name: str) -> Dict:
        """Get palette preview information."""
        palette = self.get_palette(palette_name)
        if not palette:
            return {}
        
        return {
            "name": palette["name"],
            "description": palette["description"],
            "colors": palette["colors"],
            "color_count": len(palette["colors"]),
            "prompt_addition": palette.get("prompt_addition", ""),
            "negative_addition": palette.get("negative_addition", "")
        }

    def extract_palette_from_image(self, image_path: str, num_colors: int = 5, method: str = "kmeans") -> Dict:
        """
        Extract color palette from an image.
        
        Args:
            image_path: Path to the image file
            num_colors: Number of colors to extract
            method: Extraction method ('kmeans', 'dominant', 'median_cut')
            
        Returns:
            Dictionary containing palette information
        """
        try:
            # Load image
            image = Image.open(image_path)
            image = image.convert('RGB')
            
            # Resize for faster processing
            image_small = image.resize((150, 150), Image.Resampling.LANCZOS)
            img_array = np.array(image_small)
            
            if method == "kmeans":
                colors = self._extract_kmeans_palette(img_array, num_colors)
            elif method == "dominant":
                colors = self._extract_dominant_colors(img_array, num_colors)
            elif method == "median_cut":
                colors = self._extract_median_cut_palette(img_array, num_colors)
            else:
                colors = self._extract_kmeans_palette(img_array, num_colors)
            
            # Convert to hex format
            hex_colors = [self._rgb_to_hex(color) for color in colors]
            
            # Generate palette name from image filename
            palette_name = Path(image_path).stem + "_extracted"
            
            # Analyze palette characteristics
            palette_analysis = self._analyze_palette(colors)
            
            return {
                "name": palette_name,
                "description": f"Extracted from {Path(image_path).name} using {method} method",
                "colors": hex_colors,
                "source_image": image_path,
                "extraction_method": method,
                "num_colors": num_colors,
                "analysis": palette_analysis,
                "prompt_addition": self._generate_prompt_from_palette(colors),
                "negative_addition": self._generate_negative_from_palette(colors)
            }
            
        except Exception as e:
            print(f"Error extracting palette from image: {e}")
            return {}
    
    def _extract_kmeans_palette(self, img_array: np.ndarray, num_colors: int) -> List[Tuple[int, int, int]]:
        """Extract palette using K-means clustering."""
        # Reshape image to 2D array of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Sort by frequency (most common first)
        labels = kmeans.labels_
        color_counts = np.bincount(labels)
        sorted_indices = np.argsort(color_counts)[::-1]
        colors = colors[sorted_indices]
        
        return [tuple(color) for color in colors]
    
    def _extract_dominant_colors(self, img_array: np.ndarray, num_colors: int) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using histogram analysis."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Create histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Find peaks in histograms
        colors = []
        for i in range(num_colors):
            # Find dominant hue
            h_peak = np.argmax(h_hist)
            h_hist[h_peak] = 0  # Remove peak for next iteration
            
            # Find corresponding saturation and value
            s_peak = np.argmax(s_hist)
            v_peak = np.argmax(v_hist)
            
            # Convert back to RGB
            hsv_color = np.array([[[h_peak, s_peak, v_peak]]], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
            
            colors.append(tuple(rgb_color))
        
        return colors
    
    def _extract_median_cut_palette(self, img_array: np.ndarray, num_colors: int) -> List[Tuple[int, int, int]]:
        """Extract palette using median cut algorithm."""
        pixels = img_array.reshape(-1, 3)
        
        def median_cut(pixels, depth):
            if depth == 0 or len(pixels) == 0:
                return [np.mean(pixels, axis=0).astype(int)]
            
            # Find the channel with the greatest range
            ranges = np.max(pixels, axis=0) - np.min(pixels, axis=0)
            channel = np.argmax(ranges)
            
            # Sort pixels by the channel with greatest range
            sorted_pixels = pixels[pixels[:, channel].argsort()]
            
            # Split at median
            median_idx = len(sorted_pixels) // 2
            left_pixels = sorted_pixels[:median_idx]
            right_pixels = sorted_pixels[median_idx:]
            
            # Recursively process both halves
            left_colors = median_cut(left_pixels, depth - 1)
            right_colors = median_cut(right_pixels, depth - 1)
            
            return left_colors + right_colors
        
        # Calculate depth needed for desired number of colors
        depth = int(np.log2(num_colors))
        colors = median_cut(pixels, depth)
        
        # Ensure we get exactly num_colors
        while len(colors) < num_colors:
            colors.append(colors[-1])  # Duplicate last color if needed
        
        return [tuple(color) for color in colors[:num_colors]]
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color string."""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def _analyze_palette(self, colors: List[Tuple[int, int, int]]) -> Dict:
        """Analyze palette characteristics."""
        if not colors:
            return {}
        
        # Convert to HSV for analysis
        hsv_colors = []
        for color in colors:
            hsv = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
            hsv_colors.append(hsv)
        
        # Calculate characteristics
        hues = [hsv[0] for hsv in hsv_colors]
        saturations = [hsv[1] for hsv in hsv_colors]
        values = [hsv[2] for hsv in hsv_colors]
        
        # Determine palette type
        avg_saturation = np.mean(saturations)
        avg_value = np.mean(values)
        hue_range = max(hues) - min(hues)
        
        if avg_saturation < 0.3:
            palette_type = "muted"
        elif avg_saturation > 0.7:
            palette_type = "vibrant"
        else:
            palette_type = "balanced"
        
        if avg_value < 0.4:
            brightness = "dark"
        elif avg_value > 0.7:
            brightness = "bright"
        else:
            brightness = "medium"
        
        if hue_range < 0.2:
            color_scheme = "monochromatic"
        elif hue_range < 0.4:
            color_scheme = "analogous"
        elif 0.4 <= hue_range <= 0.6:
            color_scheme = "complementary"
        else:
            color_scheme = "diverse"
        
        return {
            "type": palette_type,
            "brightness": brightness,
            "color_scheme": color_scheme,
            "avg_saturation": avg_saturation,
            "avg_value": avg_value,
            "hue_range": hue_range
        }
    
    def _generate_prompt_from_palette(self, colors: List[Tuple[int, int, int]]) -> str:
        """Generate prompt addition based on palette analysis."""
        analysis = self._analyze_palette(colors)
        
        prompt_parts = []
        
        # Add brightness description
        if analysis.get("brightness") == "bright":
            prompt_parts.append("bright lighting, vibrant atmosphere")
        elif analysis.get("brightness") == "dark":
            prompt_parts.append("moody lighting, dramatic atmosphere")
        else:
            prompt_parts.append("balanced lighting, natural atmosphere")
        
        # Add saturation description
        if analysis.get("type") == "vibrant":
            prompt_parts.append("saturated colors, bold tones")
        elif analysis.get("type") == "muted":
            prompt_parts.append("soft colors, muted tones")
        else:
            prompt_parts.append("natural colors, balanced tones")
        
        # Add color scheme description
        if analysis.get("color_scheme") == "monochromatic":
            prompt_parts.append("monochromatic color palette")
        elif analysis.get("color_scheme") == "analogous":
            prompt_parts.append("harmonious color palette")
        elif analysis.get("color_scheme") == "complementary":
            prompt_parts.append("complementary color palette")
        else:
            prompt_parts.append("diverse color palette")
        
        return ", ".join(prompt_parts)
    
    def _generate_negative_from_palette(self, colors: List[Tuple[int, int, int]]) -> str:
        """Generate negative prompt based on palette analysis."""
        analysis = self._analyze_palette(colors)
        
        negative_parts = []
        
        # Add opposites of palette characteristics
        if analysis.get("brightness") == "bright":
            negative_parts.append("dark, shadowy")
        elif analysis.get("brightness") == "dark":
            negative_parts.append("bright, overexposed")
        
        if analysis.get("type") == "vibrant":
            negative_parts.append("muted, dull")
        elif analysis.get("type") == "muted":
            negative_parts.append("oversaturated, artificial")
        
        return ", ".join(negative_parts)
    
    def add_palette_from_image(self, image_path: str, palette_name: str = None, 
                              num_colors: int = 5, method: str = "kmeans") -> bool:
        """
        Extract palette from image and add it to custom palettes.
        
        Args:
            image_path: Path to the image file
            palette_name: Custom name for the palette (optional)
            num_colors: Number of colors to extract
            method: Extraction method
            
        Returns:
            True if successful, False otherwise
        """
        palette_data = self.extract_palette_from_image(image_path, num_colors, method)
        
        if not palette_data:
            return False
        
        # Use custom name if provided
        if palette_name:
            palette_data["name"] = palette_name
        
        # Add to custom palettes
        self.custom_palettes[palette_data["name"]] = palette_data
        self._save_custom_palettes()
        
        return True


# Global instance
palette_manager = ColorPaletteManager()


def get_available_palettes() -> Dict[str, List[str]]:
    """Get all available palettes."""
    return palette_manager.list_palettes()


def apply_palette_to_prompt(palette_name: str, base_prompt: str = "") -> Tuple[str, str]:
    """Apply a color palette to enhance prompts."""
    return palette_manager.get_palette_prompt(palette_name, base_prompt)


def add_custom_palette(name: str, colors: List[str], description: str = "", 
                      prompt_addition: str = "", negative_addition: str = ""):
    """Add a custom color palette."""
    palette_manager.add_custom_palette(name, colors, description, prompt_addition, negative_addition)


def preview_palette(palette_name: str) -> Dict:
    """Preview a color palette."""
    return palette_manager.preview_palette(palette_name)


def extract_palette_from_image(image_path: str, num_colors: int = 5, method: str = "kmeans") -> Dict:
    """Extract color palette from an image."""
    return palette_manager.extract_palette_from_image(image_path, num_colors, method)


def add_palette_from_image(image_path: str, palette_name: str = None, 
                          num_colors: int = 5, method: str = "kmeans") -> bool:
    """Extract palette from image and add it to custom palettes."""
    return palette_manager.add_palette_from_image(image_path, palette_name, num_colors, method)


if __name__ == "__main__":
    # Example usage
    print("🎨 Available Color Palettes:")
    palettes = get_available_palettes()
    
    print("\n📋 Predefined Palettes:")
    for palette in palettes["predefined"]:
        preview = preview_palette(palette)
        print(f"  • {preview['name']}: {preview['description']}")
    
    print("\n💾 Custom Palettes:")
    for palette in palettes["custom"]:
        preview = preview_palette(palette)
        print(f"  • {preview['name']}: {preview['description']}")
    
    # Example: Add a custom palette
    print("\n➕ Adding example custom palette...")
    add_custom_palette(
        name="sunset_golden",
        colors=["#FF6B35", "#F7931E", "#FFD23F", "#FFEAA7", "#DDA0DD"],
        description="Golden sunset colors",
        prompt_addition="golden sunset lighting, warm orange tones, romantic atmosphere",
        negative_addition="cool colors, blue lighting, cold atmosphere"
    )
    
    print("✅ Custom palette added!")
