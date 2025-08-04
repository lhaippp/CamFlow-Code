#!/usr/bin/env python3
"""
CamFlow Blind Inference - Camera Motion Estimation

Estimates 2D camera motion between two images using hybrid motion basis.
Outputs optical flow fields and creates visualization GIF showing warping results.

Usage: python blind_inference.py
- Place your images as data/test_imgs/img1.png and img2.png
- Results saved as result.gif and flow data in results/ directory

Following KISS, YAGNI and SOLID principles.
"""

import os
import cv2
import torch
import numpy as np
import logging
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from common.manager import Manager
from dataset.data_loader import BlindInference
from model.net import fetch_net
from model.utils import get_warp_flow
from torch.utils.data import DataLoader

def load_images(image_paths):
    """Load two images from specified paths and track orientation changes."""
    if isinstance(image_paths, str):
        # Legacy: directory path, assume img1.png and img2.png
        img_dir = image_paths
        paths = [os.path.join(img_dir, f'img{i}.png') for i in [1, 2]]
    else:
        # New: direct image paths
        paths = image_paths
    
    images = []
    is_rotated = []  # Track if each image was rotated
    
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Check if image is portrait and needs rotation
        was_rotated = img_rgb.shape[0] > img_rgb.shape[1]
        is_rotated.append(was_rotated)
        
        images.append(img_rgb)
    
    return images, is_rotated

def save_results(data_batch, output, is_rotated=[False, False], output_dir="results"):
    """Save inference results including flows and visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    rgb_data = data_batch['imgs_rgb_full'][0].cpu().numpy()
    img1 = np.clip(rgb_data[:3].transpose(1, 2, 0), 0, 1)
    img2 = np.clip(rgb_data[3:].transpose(1, 2, 0), 0, 1)
    flow_f = output['flow_f'][0].cpu().numpy()
    flow_b = output['flow_b'][0].cpu().numpy()
    
    # Function to rotate image back if it was originally portrait
    def rotate_back_if_needed(img, was_rotated):
        if was_rotated:
            # Convert to uint8 for cv2 operations
            img_uint8 = (img * 255).astype(np.uint8)
            # Rotate counterclockwise to restore original orientation
            img_rotated = cv2.rotate(img_uint8, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return img_rotated.astype(np.float32) / 255.0
        return img
    
    # Function to rotate flow field back if needed
    def rotate_flow_back_if_needed(flow, was_rotated):
        if was_rotated:
            # Rotate flow field counterclockwise and swap x,y components
            flow_rotated = cv2.rotate(flow, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # When rotating 90° counterclockwise: new_x = -old_y, new_y = old_x
            flow_rotated = np.stack([-flow_rotated[..., 1], flow_rotated[..., 0]], axis=-1)
            return flow_rotated
        return flow
    
    # Rotate images back to original orientation if needed
    img1_final = rotate_back_if_needed(img1, is_rotated[0])
    img2_final = rotate_back_if_needed(img2, is_rotated[1])
    
    # Rotate flow fields back if needed
    flow_f_final = rotate_flow_back_if_needed(flow_f, is_rotated[1])  # Forward flow follows img2
    flow_b_final = rotate_flow_back_if_needed(flow_b, is_rotated[0])  # Backward flow follows img1
    
    # Save flow fields as numpy arrays
    np.save(os.path.join(output_dir, 'flow_forward.npy'), flow_f_final)
    np.save(os.path.join(output_dir, 'flow_backward.npy'), flow_b_final)
    
    # Save individual images
    to_pil = lambda x: Image.fromarray((x * 255).astype(np.uint8))
    to_pil(img1_final).save(os.path.join(output_dir, 'img1.png'))
    to_pil(img2_final).save(os.path.join(output_dir, 'img2.png'))
    
    # Save warped images - need to handle rotation for warping
    def warp_image(img, flow, was_rotated_img, was_rotated_flow):
        # For warping, we work in the rotated space (landscape) and then rotate back
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        flow_t = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
        warped = get_warp_flow(img_t, flow_t)[0].permute(1, 2, 0).clamp(0, 1).numpy()
        
        # Rotate back if original image was portrait
        return rotate_back_if_needed(warped, was_rotated_img)
    
    img2_warped = warp_image(img2, flow_f, is_rotated[1], is_rotated[1])
    img1_warped = warp_image(img1, flow_b, is_rotated[0], is_rotated[0])
    
    to_pil(img2_warped).save(os.path.join(output_dir, 'img2_warped.png'))
    to_pil(img1_warped).save(os.path.join(output_dir, 'img1_warped.png'))
    
    return output_dir

def create_gif(data_batch, output, is_rotated=[False, False], path="result.gif"):
    """Create animated GIF showing camera motion estimation results."""
    # Extract data
    rgb_data = data_batch['imgs_rgb_full'][0].cpu().numpy()
    img1 = np.clip(rgb_data[:3].transpose(1, 2, 0), 0, 1)
    img2 = np.clip(rgb_data[3:].transpose(1, 2, 0), 0, 1)
    flow_f = output['flow_f'][0].cpu().numpy()
    flow_b = output['flow_b'][0].cpu().numpy()
    
    # Function to rotate image back if it was originally portrait
    def rotate_back_if_needed(img, was_rotated):
        if was_rotated:
            # Convert to uint8 for cv2 operations
            img_uint8 = (img * 255).astype(np.uint8)
            # Rotate counterclockwise to restore original orientation
            img_rotated = cv2.rotate(img_uint8, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return img_rotated.astype(np.float32) / 255.0
        return img
    
    # Function to rotate flow field back if needed
    def rotate_flow_back_if_needed(flow, was_rotated):
        if was_rotated:
            # Rotate flow field counterclockwise and swap x,y components
            flow_rotated = cv2.rotate(flow, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # When rotating 90° counterclockwise: new_x = -old_y, new_y = old_x
            flow_rotated = np.stack([-flow_rotated[..., 1], flow_rotated[..., 0]], axis=-1)
            return flow_rotated
        return flow
    
    # Rotate images back to original orientation if needed
    img1_final = rotate_back_if_needed(img1, is_rotated[0])
    img2_final = rotate_back_if_needed(img2, is_rotated[1])
    
    # Convert to PIL
    to_pil = lambda x: Image.fromarray((x * 255).astype(np.uint8))
    img1_pil, img2_pil = to_pil(img1_final), to_pil(img2_final)
    
    # Warp images using estimated flows
    def warp_image(img, flow, was_rotated_img, was_rotated_flow):
        # For warping, we work in the rotated space (landscape) and then rotate back
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        flow_t = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
        warped = get_warp_flow(img_t, flow_t)[0].permute(1, 2, 0).clamp(0, 1).numpy()
        
        # Rotate back if original image was portrait
        return rotate_back_if_needed(warped, was_rotated_img)
    
    img2_warped = to_pil(warp_image(img2, flow_f, is_rotated[1], is_rotated[1]))
    img1_warped = to_pil(warp_image(img1, flow_b, is_rotated[0], is_rotated[0]))
    
    # Create animation frames
    H, W = img1_final.shape[:2]
    frame1 = Image.new('RGB', (W * 3, H))
    frame2 = Image.new('RGB', (W * 3, H))
    
    # Frame 1: Original sequence [img1 | img1 | img2]
    for i, img in enumerate([img1_pil, img1_pil, img2_pil]):
        frame1.paste(img, (i * W, 0))
    
    # Frame 2: Warped results [img2 | img2→img1 | img1→img2]
    for i, img in enumerate([img2_pil, img2_warped, img1_warped]):
        frame2.paste(img, (i * W, 0))
    
    # Save animated GIF
    frame1.save(path, save_all=True, append_images=[frame2], duration=150, loop=0)
    return path

class InferenceConfig:
    """Simplified configuration for inference."""
    def __init__(self):
        # Core model params (from data/params.json)
        self.net_type = "CamFlow"
        self.num_basis = 24
        self.crop_size = [320, 576]
        self.ori_size = [360, 640]
        self.generate_size = 256
        self.rho = 16
        
        # Model architecture
        self.in_channels = 2
        self.patch_size = 4
        self.in_chans = 2
        self.embed_dim = 24
        self.depths = [2, 4, 6]
        self.layer_depth = [3, 2, 1]
        self.num_heads = [3, 12, 24]
        self.num_decoder_heads = 4
        self.num_decoder_layers = 4
        self.window_size = 8
        self.mlp_ratio = 3
        
        # Required flags
        self.qkv_bias = True
        self.qk_scale = None
        self.drop_rate = 0
        self.drop_path_rate = 0
        self.attn_drop_rate = 0
        self.ape = False
        self.patch_norm = True
        self.use_checkpoint = False
        self.shift = 4
        self.cls_weight = 0.1
        self.dynamic_apha = False
        self.mask_use_fea = True
        self.mk_weight = 0.2
        self.h_weight = 1
        self.use_open = True
        self.pretrain_phase = True
        
        # Runtime - use relative data paths
        self.cuda = torch.cuda.is_available()
        self.eval_type = []
        self.model_dir = 'data'
        self.restore_file = 'data/ckpt.pth'  # Default path, can be overridden
        self.only_weights = True
        self.basis_dir = 'data/'
        self.db_path = "data"
        self.trainset = "dgm"
        self.train_batch_size = 8
        self.eval_batch_size = 8
        self.num_workers = 0
        self.imgs = None

def run_inference(images, is_rotated, model_path='data/ckpt.pth'):
    """Run inference on image pair."""
    # Setup config
    config = InferenceConfig()
    config.imgs = images
    config.restore_file = model_path
    
    # Create dataset and loader
    dataset = BlindInference(config, phase='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=config.cuda)
    
    # Setup model
    model = fetch_net(config)
    accelerator = Accelerator(split_batches=True, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    model = accelerator.prepare(model)
    loader = accelerator.prepare(loader)
    
    # Load weights
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    manager = Manager(model, None, None, config, {'blind': loader}, None, logger, accelerator)
    
    if model_path and os.path.exists(model_path):
        manager.load_checkpoints()
    else:
        print("Warning: Using random weights")
    
    # Inference
    model.eval()
    with torch.no_grad():
        for batch in loader:
            return batch, model(batch), is_rotated
    
    # If we reach here, no data found
    raise RuntimeError("No data found in loader. Check image paths and data loading.")

def main():
    """CamFlow Camera Motion Estimation - Main Entry Point"""
    import argparse
    parser = argparse.ArgumentParser(
        description='CamFlow: Estimate 2D camera motion between two images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python blind_inference.py                    # Use default paths
  python blind_inference.py --model_path custom.pth  # Custom model
  python blind_inference.py --imgs_dir /path/to/images  # Custom image directory

Output:
  - result.gif: Animated visualization of motion estimation
  - results/: Directory with detailed outputs (flows, warped images)
        """
    )
    parser.add_argument('--model_path', default='data/ckpt.pth', 
                       help='Model weights path (default: data/ckpt.pth)')
    parser.add_argument('--imgs_dir', default='data/test_imgs',
                       help='Directory containing img1.png and img2.png (default: data/test_imgs)')
    parser.add_argument('--image_paths', nargs=2, metavar=('IMG1', 'IMG2'),
                       help='Direct paths to two images (overrides --imgs_dir)')
    args = parser.parse_args()
    
    print("🎥 CamFlow - Camera Motion Estimation")
    print("=" * 50)
    
    # Determine image paths
    if args.image_paths:
        image_paths = args.image_paths
        img1_path, img2_path = image_paths
    else:
        image_paths = args.imgs_dir
        img1_path = f'{args.imgs_dir}/img1.png'
        img2_path = f'{args.imgs_dir}/img2.png'
    
    # Display all parameters and paths
    print("\n⚙️  Configuration Parameters:")
    print(f"   • Model checkpoint: {args.model_path}")
    print(f"   • Motion basis file: data/basis_24.pt")
    print(f"   • Configuration file: data/params.json")
    if args.image_paths:
        print(f"   • Input images (direct paths):")
        print(f"     - Image 1: {img1_path}")
        print(f"     - Image 2: {img2_path}")
    else:
        print(f"   • Input images directory: {args.imgs_dir}/")
        print(f"     - Image 1: {img1_path}")
        print(f"     - Image 2: {img2_path}")
    print(f"   • Output directory: results/")
    print(f"   • Visualization output: result.gif")
    
    # Check file existence
    print("\n🔍 File Status Check:")
    files_to_check = [
        args.model_path,
        'data/basis_24.pt', 
        'data/params.json',
        img1_path,
        img2_path
    ]
    
    for file_path in files_to_check:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"   {status} {file_path}")
    
    print()  # Empty line for spacing
    
    try:
        # Load images - use the correct path strategy
        if args.image_paths:
            print(f"📸 Loading image pair from direct paths...")
            images, is_rotated = load_images(args.image_paths)
        else:
            print(f"📸 Loading image pair from {args.imgs_dir}/...")
            images, is_rotated = load_images(args.imgs_dir)
        print(f"   ✓ Image 1: {images[0].shape}")
        print(f"   ✓ Image 2: {images[1].shape}")
        
        # Check and notify about rotation
        if any(is_rotated):
            print("\n🔄 Image Orientation Processing:")
            if is_rotated[0]:
                print("   📱 Image 1 was portrait → automatically rotated to landscape for processing")
            if is_rotated[1]:
                print("   📱 Image 2 was portrait → automatically rotated to landscape for processing") 
            print("   ✓ All outputs will be rotated back to original orientation")
        
        # Run inference  
        print("\n🧠 Running camera motion estimation...")
        batch, output, is_rotated_output = run_inference(images, is_rotated, args.model_path)
        print("   ✓ Motion estimation complete!")
        
        # Analyze results
        print("\n📊 Motion Analysis Results:")
        flow_f = output['flow_f'][0].cpu().numpy()
        flow_b = output['flow_b'][0].cpu().numpy()
        
        # Calculate motion statistics
        flow_f_mag = np.sqrt(flow_f[..., 0]**2 + flow_f[..., 1]**2)
        flow_b_mag = np.sqrt(flow_b[..., 0]**2 + flow_b[..., 1]**2)
        
        print(f"   • Forward flow magnitude: {flow_f_mag.mean():.2f} ± {flow_f_mag.std():.2f} pixels")
        print(f"   • Backward flow magnitude: {flow_b_mag.mean():.2f} ± {flow_b_mag.std():.2f} pixels")
        print(f"   • Max motion detected: {max(flow_f_mag.max(), flow_b_mag.max()):.2f} pixels")
        
        # Save detailed results
        print("\n💾 Saving results...")
        results_dir = save_results(batch, output, is_rotated_output)
        print(f"   ✓ Detailed results saved to: {results_dir}/")
        print(f"     - flow_forward.npy: Forward optical flow field")
        print(f"     - flow_backward.npy: Backward optical flow field") 
        print(f"     - img1.png, img2.png: Original images")
        print(f"     - img1_warped.png, img2_warped.png: Motion-compensated images")
        
        # Create visualization
        print("\n🎬 Creating motion visualization...")
        gif_path = create_gif(batch, output, is_rotated_output)
        print(f"   ✓ Animation saved: {gif_path}")
        print("     - Frame 1: [Original img1 | Original img1 | Original img2]")
        print("     - Frame 2: [Original img2 | img2→img1 warped | img1→img2 warped]")
        
        print("\n✨ Camera motion estimation completed successfully!")
        print(f"   📁 Check {results_dir}/ for detailed outputs")
        print(f"   🎥 View {gif_path} for motion visualization")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.image_paths:
            print(f"💡 Make sure both image paths exist:")
            print(f"   - {args.image_paths[0]}")
            print(f"   - {args.image_paths[1]}")
        else:
            print(f"💡 Make sure {args.imgs_dir}/img1.png and {args.imgs_dir}/img2.png exist")

if __name__ == '__main__':
    main()