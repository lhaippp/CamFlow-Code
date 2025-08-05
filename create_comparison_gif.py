#!/usr/bin/env python3
"""Simplified Image Comparison GIF Generator

Following KISS, YAGNI, SOLID principles:
- Single Responsibility: Each function does one thing
- Open/Closed: Extend functionality through configuration
- Dependency Inversion: Depend on abstractions, not concretions
- YAGNI: Remove unnecessary features
- KISS: Keep it simple and understandable
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional


class ImageComparator:
    """Image comparator class - Single Responsibility Principle"""
    
    def __init__(self, comparison_folder: str = "qualitative_comparisons"):
        self.comparison_folder = Path(comparison_folder)
        self.original_folder = self.comparison_folder / "I"
        self.output_folder = self.comparison_folder / "comparison_gifs"
        
    def get_comparison_methods(self) -> List[str]:
        """Get comparison method folders"""
        if not self.comparison_folder.exists():
            raise FileNotFoundError(f"Comparison folder does not exist: {self.comparison_folder}")
            
        methods = []
        for folder in self.comparison_folder.iterdir():
            if folder.is_dir() and folder.name != "I" and not folder.name.startswith('.'):
                if any(folder.glob("*.png")):
                    methods.append(folder.name)
        return sorted(methods)
    
    def find_matching_image(self, method: str, filename: str) -> Optional[Path]:
        """Find matching image file"""
        method_path = self.comparison_folder / method
        
        # Direct match
        direct_path = method_path / filename
        if direct_path.exists():
            return direct_path
        
        # CamFlow special format handling
        if method == "CamFlow":
            parts = filename.split('_')
            if len(parts) >= 2:
                prefix = parts[0]
                try:
                    number = int(parts[1].split('.')[0])
                    new_filename = f"{prefix}_{number:04d}.png"
                    new_path = method_path / new_filename
                    if new_path.exists():
                        return new_path
                except ValueError:
                    pass
        
        return None
    
    def get_image_files(self) -> List[str]:
        """Get original image file list"""
        if not self.original_folder.exists():
            raise FileNotFoundError(f"Original image folder does not exist: {self.original_folder}")
        
        return sorted([f.name for f in self.original_folder.glob("*.png")])


class ImageProcessor:
    """Image processor class - Single Responsibility Principle"""
    
    @staticmethod
    def swap_columns(image: Image.Image) -> Image.Image:
        """Swap left and right columns of image"""
        width, height = image.size
        half_width = width // 2
        
        left = image.crop((0, 0, half_width, height))
        right = image.crop((half_width, 0, width, height))
        
        result = Image.new('RGB', (width, height))
        result.paste(right, (0, 0))
        result.paste(left, (half_width, 0))
        
        return result
    
    @staticmethod
    def resize_keep_aspect(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image while keeping aspect ratio"""
        original_size = image.size
        ratio = min(target_size[0]/original_size[0], target_size[1]/original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        background = Image.new('RGB', target_size, (255, 255, 255))
        
        offset = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
        background.paste(resized, offset)
        
        return background
    
    @staticmethod
    def add_label(image: Image.Image, text: str) -> Image.Image:
        """Add label at the bottom of image"""
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (image.width - text_width) // 2
        y = image.height - text_height - 5
        
        draw.rectangle([x-3, y-2, x+text_width+3, y+text_height+2], fill=(0, 0, 0, 180))
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        return image_copy


class GridImageCreator:
    """Grid image creator class - Single Responsibility Principle"""
    
    def __init__(self, grid_size: Tuple[int, int] = (3, 4), cell_size: Tuple[int, int] = (400, 200)):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.max_cells = grid_size[0] * grid_size[1]
    
    def create_grid(self, images: List[Image.Image], labels: List[str]) -> Image.Image:
        """Create grid image"""
        rows, cols = self.grid_size
        total_width = cols * self.cell_size[0]
        total_height = rows * self.cell_size[1]
        
        grid = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        
        for i, (image, label) in enumerate(zip(images[:self.max_cells], labels[:self.max_cells])):
            row = i // cols
            col = i % cols
            
            x = col * self.cell_size[0]
            y = row * self.cell_size[1]
            
            resized = ImageProcessor.resize_keep_aspect(image, self.cell_size)
            labeled = ImageProcessor.add_label(resized, label)
            
            grid.paste(labeled, (x, y))
        
        return grid


class GifGenerator:
    """GIF generator class - Single Responsibility Principle"""
    
    def __init__(self, comparator: ImageComparator):
        self.comparator = comparator
        self.grid_creator = GridImageCreator()
    
    def create_comparison_gif(self, filename: str) -> bool:
        """Create comparison GIF for specified image"""
        methods = self.comparator.get_comparison_methods()
        
        # Limit to 12 methods
        methods = methods[:12]
        while len(methods) < 12:
            methods.append("I")
        
        # Load original image
        original_path = self.comparator.original_folder / filename
        if not original_path.exists():
            print(f"Original image does not exist: {filename}")
            return False
        
        try:
            original_image = Image.open(original_path)
        except Exception as e:
            print(f"Cannot load original image {filename}: {e}")
            return False
        
        # Prepare labels
        labels = ["Original" if method == "I" else method for method in methods]
        
        # Create first frame (all show original image)
        frame1_images = [original_image] * 12
        frame1 = self.grid_creator.create_grid(frame1_images, labels)
        
        # Create second frame (show comparison results)
        frame2_images = []
        for method in methods:
            if method == "I":
                frame2_images.append(original_image)
            else:
                method_path = self.comparator.find_matching_image(method, filename)
                if method_path:
                    try:
                        method_image = Image.open(method_path)
                        method_image = ImageProcessor.swap_columns(method_image)
                        frame2_images.append(method_image)
                    except Exception as e:
                        print(f"Cannot load {method} image: {e}")
                        frame2_images.append(original_image)
                else:
                    frame2_images.append(original_image)
        
        frame2 = self.grid_creator.create_grid(frame2_images, labels)
        
        # Save GIF
        self.comparator.output_folder.mkdir(exist_ok=True)
        gif_path = self.comparator.output_folder / f"{Path(filename).stem}_comparison.gif"
        
        frame1.save(
            gif_path,
            save_all=True,
            append_images=[frame2],
            duration=150,
            loop=0
        )
        
        print(f"Created GIF: {gif_path}")
        return True
    
    def create_all_gifs(self) -> None:
        """Create comparison GIFs for all images"""
        image_files = self.comparator.get_image_files()
        
        if not image_files:
            print("No image files found")
            return
        
        print(f"Starting to process {len(image_files)} images...")
        
        success_count = 0
        for i, filename in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {filename}")
            if self.create_comparison_gif(filename):
                success_count += 1
        
        print(f"Successfully processed {success_count}/{len(image_files)} images")
        print(f"All GIFs saved to: {self.comparator.output_folder}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate image comparison GIFs")
    parser.add_argument(
        "--folder", 
        default="qualitative_comparisons",
        help="Comparison folder path (default: qualitative_comparisons)"
    )
    parser.add_argument(
        "--image",
        help="Specify single image filename"
    )
    
    args = parser.parse_args()
    
    try:
        comparator = ImageComparator(args.folder)
        generator = GifGenerator(comparator)
        
        if args.image:
            generator.create_comparison_gif(args.image)
        else:
            generator.create_all_gifs()
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())