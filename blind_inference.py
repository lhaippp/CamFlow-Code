#!/usr/bin/env python3
"""
Blind inference script for CamFlow model.
Simplified following KISS, YAGNI and SOLID principles.
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

def load_images():
    """Load two images from data/test_imgs directory."""
    img_dir = 'data/test_imgs'
    paths = [os.path.join(img_dir, f'img{i}.png') for i in [1, 2]]
    
    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    return images

def create_gif(data_batch, output, path="result.gif"):
    """Create GIF with original and warped images."""
    # Extract images and flows
    rgb_data = data_batch['imgs_rgb_full'][0].cpu().numpy()  # [6, H, W]
    img1 = np.clip(rgb_data[:3].transpose(1, 2, 0), 0, 1)
    img2 = np.clip(rgb_data[3:].transpose(1, 2, 0), 0, 1)
    
    flow_f = output['flow_f'][0].cpu().numpy()
    flow_b = output['flow_b'][0].cpu().numpy()
    
    # Convert to PIL images
    to_pil = lambda x: Image.fromarray((x * 255).astype(np.uint8))
    img1_pil, img2_pil = to_pil(img1), to_pil(img2)
    
    # Warp images
    def warp_image(img, flow):
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        flow_t = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
        warped = get_warp_flow(img_t, flow_t)
        return warped[0].permute(1, 2, 0).clamp(0, 1).numpy()
    
    img2_warped = to_pil(warp_image(img2, flow_f))
    img1_warped = to_pil(warp_image(img1, flow_b))
    
    # Create frames
    H, W = img1.shape[:2]
    frame1 = Image.new('RGB', (W * 3, H))
    frame2 = Image.new('RGB', (W * 3, H))
    
    # Frame 1: img1 | img1 | img2
    for i, img in enumerate([img1_pil, img1_pil, img2_pil]):
        frame1.paste(img, (i * W, 0))
    
    # Frame 2: img2 | img2_warped | img1_warped
    for i, img in enumerate([img2_pil, img2_warped, img1_warped]):
        frame2.paste(img, (i * W, 0))
    
    # Save GIF
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
        self.restore_file = None
        self.only_weights = True
        self.basis_dir = 'data/'
        self.db_path = "data"
        self.trainset = "dgm"
        self.train_batch_size = 8
        self.eval_batch_size = 8
        self.num_workers = 0
        self.imgs = None

def run_inference(images, model_path='data/ckpt.pth'):
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
            return batch, model(batch)

def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description='CamFlow blind inference')
    parser.add_argument('--model_path', default='data/ckpt.pth', help='Model weights path (default: data/ckpt.pth)')
    args = parser.parse_args()
    
    try:
        # Load images
        print("Loading images from data/test_imgs...")
        images = load_images()
        print(f"Loaded: {[img.shape for img in images]}")
        
        # Run inference  
        print("Running inference...")
        batch, output = run_inference(images, args.model_path)
        
        # Show results
        print("Output shapes:")
        for k, v in output.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
        
        # Create visualization
        print("Creating GIF...")
        gif_path = create_gif(batch, output, "result.gif")
        print(f"Saved: {gif_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()