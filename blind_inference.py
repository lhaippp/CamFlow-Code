#!/usr/bin/env python3
"""
Blind inference script for CamFlow model using test_imgs.
Following KISS, YAGNI and SOLID principles.
"""

import os
import cv2
import torch
import numpy as np
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from PIL import Image
import torch.nn.functional as F

from common.utils import Params
from common.manager import Manager
from dataset.data_loader import BlindInference
from model.net import fetch_net
from model.utils import get_warp_flow
from torch.utils.data import DataLoader

def load_test_images():
    """Load images from test_imgs directory."""
    img_dir = '/mnt/exp/CamFlow-Code/test_imgs'
    img1_path = os.path.join(img_dir, 'img1.png')
    img2_path = os.path.join(img_dir, 'img2.png')
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load images from test_imgs")
    
    # Convert BGR to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) 
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    return img1, img2

def create_visualization_gif(data_batch, output, save_path="result.gif"):
    """Create GIF visualization with original and warped images."""
    # Extract RGB images (they are concatenated as [img1, img2] in channel dimension)
    imgs_rgb_full = data_batch['imgs_rgb_full'][0].cpu().numpy()  # [6, H, W]
    
    # The concatenated image is [6, H, W] where first 3 channels are img1, last 3 are img2
    if imgs_rgb_full.shape[0] == 6:
        img1_rgb = imgs_rgb_full[:3].transpose(1, 2, 0)  # [H, W, 3]  
        img2_rgb = imgs_rgb_full[3:].transpose(1, 2, 0)  # [H, W, 3]
        H, W = img1_rgb.shape[:2]
    else:
        # Fall back to width splitting
        H, W_total = imgs_rgb_full.shape[1], imgs_rgb_full.shape[2]
        W = W_total // 2
        img1_rgb = imgs_rgb_full[:, :, :W].transpose(1, 2, 0)  # [H, W, 3]
        img2_rgb = imgs_rgb_full[:, :, W:].transpose(1, 2, 0)  # [H, W, 3]
    
    # Extract flows
    flow_f = output['flow_f'][0].cpu().numpy()  # [H, W, 2]
    flow_b = output['flow_b'][0].cpu().numpy()  # [H, W, 2]
    
    # Clamp images to [0, 1] and convert to PIL format
    img1_rgb = np.clip(img1_rgb, 0, 1)
    img2_rgb = np.clip(img2_rgb, 0, 1)
    
    img1_pil = Image.fromarray((img1_rgb * 255).astype(np.uint8))
    img2_pil = Image.fromarray((img2_rgb * 255).astype(np.uint8))
    
    # Create warped images using the flows
    img1_torch = torch.from_numpy(img1_rgb).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
    img2_torch = torch.from_numpy(img2_rgb).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
    
    # flow_f is for img2, flow_b is for img1
    # Warp img2 with forward flow 
    flow_f_torch = torch.from_numpy(flow_f).permute(2, 0, 1).unsqueeze(0).float()
    img2_warped = get_warp_flow(img2_torch, flow_f_torch)
    img2_warped_np = img2_warped[0].permute(1, 2, 0).clamp(0, 1).numpy()
    img2_warped_pil = Image.fromarray((img2_warped_np * 255).astype(np.uint8))
    
    # Warp img1 with backward flow
    flow_b_torch = torch.from_numpy(flow_b).permute(2, 0, 1).unsqueeze(0).float()
    img1_warped = get_warp_flow(img1_torch, flow_b_torch)
    img1_warped_np = img1_warped[0].permute(1, 2, 0).clamp(0, 1).numpy()
    img1_warped_pil = Image.fromarray((img1_warped_np * 255).astype(np.uint8))
    
    # Create composite frames
    # Frame 1: img1 | img1 | img2
    frame1 = Image.new('RGB', (W * 3, H))
    frame1.paste(img1_pil, (0, 0))        # img1
    frame1.paste(img1_pil, (W, 0))        # img1 (duplicate)
    frame1.paste(img2_pil, (W*2, 0))      # img2
    
    # Frame 2: img2 | img2_warped | img1_warped  
    frame2 = Image.new('RGB', (W * 3, H))
    frame2.paste(img2_pil, (0, 0))        # img2
    frame2.paste(img2_warped_pil, (W, 0)) # img2 warped with flow_f
    frame2.paste(img1_warped_pil, (W*2, 0)) # img1 warped with flow_b
    
    # Save as GIF
    frames = [frame1, frame2]
    frames[0].save(save_path, save_all=True, append_images=frames[1:], 
                   duration=150, loop=0)  # 1.5 seconds per frame
    
    print(f"Visualization saved to {save_path}")
    return save_path

def setup_params():
    """Setup parameters for inference based on data/params.json."""
    class SimpleParams:
        def __init__(self):
            # From data/params.json
            self.crop_size = [320, 576]
            self.ori_size = [360, 640]
            self.generate_size = 256
            self.train_batch_size = 8
            self.eval_batch_size = 8
            self.num_workers = 0  # Set to 0 for inference
            self.rho = 16
            
            # Model architecture
            self.net_type = "CamFlow"
            self.num_basis = 24
            self.in_channels = 2
            self.patch_size = 4
            self.in_chans = 2
            self.embed_dim = 24
            
            # Swin Transformer params
            self.depths = [2, 4, 6]
            self.layer_depth = [3, 2, 1]
            self.num_heads = [3, 12, 24]
            self.num_decoder_heads = 4
            self.num_decoder_layers = 4
            self.window_size = 8
            self.mlp_ratio = 3
            self.qkv_bias = True
            self.qk_scale = None
            self.drop_rate = 0
            self.drop_path_rate = 0
            self.attn_drop_rate = 0
            self.ape = False
            self.patch_norm = True
            self.use_checkpoint = False
            
            # Additional params
            self.shift = 4
            self.cls_weight = 0.1
            self.dynamic_apha = False
            self.mask_use_fea = True
            self.mk_weight = 0.2
            self.h_weight = 1
            self.use_open = True
            self.pretrain_phase = True
            
            # Runtime params
            self.cuda = torch.cuda.is_available()
            self.eval_type = []
            self.model_dir = '.'
            self.restore_file = None
            self.only_weights = True
            self.basis_dir = './'
            self.db_path = "."
            self.trainset = "dgm"
            # Placeholder for images - will be set later
            self.imgs = None
    
    return SimpleParams()

def run_inference(img1, img2, model_path=None):
    """Run blind inference on two images."""
    # Setup
    params = setup_params()
    params.imgs = [img1, img2]
    if model_path:
        params.restore_file = model_path
    
    # Create blind dataset and dataloader directly
    blind_dataset = BlindInference(params, phase='test')
    blind_loader = DataLoader(
        blind_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=params.cuda,
    )
    
    # Create model
    model = fetch_net(params)
    
    # Setup accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=True, kwargs_handlers=[kwargs])
    model = accelerator.prepare(model)
    blind_loader = accelerator.prepare(blind_loader)
    
    # Create manager and load weights
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    manager = Manager(
        model=model,
        optimizer=None,
        scheduler=None, 
        params=params,
        dataloaders={'blind': blind_loader},
        writer=None,
        logger=logger,
        accelerator=accelerator
    )
    
    if model_path and os.path.exists(model_path):
        manager.load_checkpoints()
    else:
        print("Warning: Model file not found, using random weights")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        for data_batch in blind_loader:
            output = model(data_batch)
            return data_batch, output

def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Blind inference on test images')
    parser.add_argument('--model_path', help='Path to model weights file')
    args = parser.parse_args()
    
    print("Loading test images...")
    img1, img2 = load_test_images()
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    
    if args.model_path and not os.path.exists(args.model_path):
        print(f"Warning: Model file {args.model_path} not found")
        print("Will proceed with random weights for testing")
    
    print("Running inference...")
    try:
        data_batch, output = run_inference(img1, img2, args.model_path)
        
        print("Inference complete!")
        print("Output keys:", list(output.keys()))
        
        # Print output shapes
        for key, value in output.items():
            if hasattr(value, 'shape'):
                print(f"{key}: {value.shape}")
        
        # Create visualization
        print("Creating visualization...")
        gif_path = create_visualization_gif(data_batch, output, "blind_inference_result.gif")
        print(f"Results saved to {gif_path}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        print("This is expected if model weights are not available")

if __name__ == '__main__':
    main()