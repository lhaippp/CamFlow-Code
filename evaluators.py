"""
Evaluator classes for CamFlow model.
Following SOLID principles with single responsibility for each evaluator.
"""

import os
import torch
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod

import pyiqa
from model.flow_utils import flow_error_avg
from model.utils import get_warp_flow
from common.iqa_utils import compute_masked_metrics

class BaseEvaluator(ABC):
    """Base evaluator class defining interface."""
    
    def __init__(self, manager):
        self.manager = manager
        self.model = manager.model
        self.accelerator = manager.accelerator
        self.logger = manager.logger
    
    @abstractmethod
    def evaluate(self):
        """Run evaluation."""
        pass

class GHOFEvaluator(BaseEvaluator):
    """Evaluator for GHOF metrics."""
    
    def __init__(self, manager):
        super().__init__(manager)
        self.categories = ['RE', 'Fog', 'Dark', 'Rain', 'SNOW']
        self.outputs = []  # Store model outputs
        
    def compute_test_metrics(self, data, endpoints):
        """Compute EPE and PME metrics."""
        flow_fw = endpoints["flow_f"]
        flow_bw = endpoints["flow_b"]
        
        flow_fw = flow_fw.permute(0, 3, 1, 2)
        flow_bw = flow_bw.permute(0, 3, 1, 2)
        
        gt_flow = data["gt_flow"]
        homo_gt = data["homo_field"]
        mask = data["mask"]
        bs, _, h, _ = gt_flow.shape
        
        # Flow metrics
        epe_flow = flow_error_avg(flow_fw * mask, gt_flow * mask)[0]
        
        # Homography metrics (focus on background motion)
        epe_homo, pck1_homo, pck5_homo = flow_error_avg(
            flow_fw[:, :, :h // 2], homo_gt[:, :, :h // 2])
            
        return epe_flow, epe_homo
    
    def evaluate(self):
        """Run GHOF evaluation."""
        self.model.eval()
        torch.cuda.empty_cache()
        
        # Metric containers
        epe_dict = {cat: [] for cat in self.categories}
        pme_dict = {cat: [] for cat in self.categories}
        
        with torch.no_grad():
            if self.manager.dataloaders["ghof"] is not None:
                for data_batch in self.manager.dataloaders["ghof"]:
                    # Get predictions
                    output = self.model(data_batch)
                    self.outputs.append((data_batch, output))  # Store results for IQA
                    video_names = data_batch["video_names"]
                    
                    # Compute metrics
                    epe_flow, epe_homo = self.compute_test_metrics(data_batch, output)
                    
                    # Store results by category
                    for i, category in enumerate(video_names):
                        if category in epe_dict:
                            epe_dict[category].append(epe_flow[i].item())
                            pme_dict[category].append(epe_homo[i].item())
        
        # Gather and average results
        metrics = {}
        for cat in self.categories:
            # Compute mean for each category
            epe_values = torch.tensor(epe_dict[cat], device=self.accelerator.device)
            pme_values = torch.tensor(pme_dict[cat], device=self.accelerator.device)
            
            epe_sum = self.accelerator.gather_for_metrics(epe_values).sum()
            epe_count = self.accelerator.gather_for_metrics(epe_values).numel()
            pme_sum = self.accelerator.gather_for_metrics(pme_values).sum()
            pme_count = self.accelerator.gather_for_metrics(pme_values).numel()
            
            metrics[f"EPE_{cat}"] = epe_sum / epe_count if epe_count > 0 else torch.tensor(0.0)
            metrics[f"PME_{cat}"] = pme_sum / pme_count if pme_count > 0 else torch.tensor(0.0)
            
            if self.accelerator.is_main_process:
                self.logger.info(f"{cat} samples - EPE: {epe_count}, PME: {pme_count}")
        
        # Compute overall averages
        metrics["EPE_avg"] = sum(metrics[f"EPE_{cat}"] for cat in self.categories) / len(self.categories)
        metrics["PME_avg"] = sum(metrics[f"PME_{cat}"] for cat in self.categories) / len(self.categories)
        
        if self.accelerator.is_main_process:
            self._print_summary(metrics)
            
        # Save best models
        if self.accelerator.is_main_process:
            self._save_best_models(metrics)
            
        return metrics["EPE_avg"], metrics["PME_avg"]
    
    def _print_summary(self, metrics):
        """Print evaluation summary."""
        self.logger.info("\n=== GHOF Metrics Summary ===")
        
        # Print per-category metrics
        for cat in self.categories:
            self.logger.info(
                f"{cat}: EPE={metrics[f'EPE_{cat}']:.4f}, PME={metrics[f'PME_{cat}']:.4f}"
            )
        
        # Print overall averages
        self.logger.info(
            f"\nOverall: EPE_avg={metrics['EPE_avg']:.4f}, PME_avg={metrics['PME_avg']:.4f}"
        )
        self.logger.info("==========================\n")
    
    def _save_best_models(self, metrics):
        """Save models with best metrics."""
        # Save best EPE model
        if not hasattr(self.manager, 'best_EPE_avg') or metrics["EPE_avg"] < self.manager.best_EPE_avg:
            self.manager.best_EPE_avg = metrics["EPE_avg"]
            
            # Create state dict without optimizer
            state = {
                "epoch": self.manager.epoch,
                "state_dict": self.manager.model.state_dict(),
                "best_EPE_avg": metrics["EPE_avg"].item()
            }
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                self.manager.params.model_dir,
                f"best_epe_{metrics['EPE_avg']:.4f}.pth"
            )
            torch.save(state, checkpoint_path)
            self.logger.info(f"Saved best EPE_avg checkpoint with score {metrics['EPE_avg']:.4f}")
            
        # Save best PME model
        if not hasattr(self.manager, 'best_PME_avg') or metrics["PME_avg"] < self.manager.best_PME_avg:
            self.manager.best_PME_avg = metrics["PME_avg"]
            
            # Create state dict without optimizer
            state = {
                "epoch": self.manager.epoch,
                "state_dict": self.manager.model.state_dict(),
                "best_PME_avg": metrics["PME_avg"].item()
            }
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                self.manager.params.model_dir,
                f"best_pme_{metrics['PME_avg']:.4f}.pth"
            )
            torch.save(state, checkpoint_path)
            self.logger.info(f"Saved best PME_avg checkpoint with score {metrics['PME_avg']:.4f}")

class IQAEvaluator(BaseEvaluator):
    """Evaluator for Image Quality Assessment metrics."""
    
    def __init__(self, manager):
        super().__init__(manager)
        self.scene_ranges = {
            "reg": (0, 52),
            "fog": (53, 101),
            "dark": (102, 156),
            "rain": (157, 203),
            "snow": (204, 253)
        }
        self.iqa_metrics = self._init_iqa_metrics()
        
    def _init_iqa_metrics(self):
        """Initialize IQA metrics."""
        try:
            metrics = {
                'PSNR': pyiqa.create_metric('psnr'),
                'SSIM': pyiqa.create_metric('ssim'),
                'LPIPS': pyiqa.create_metric('lpips'),
            }
            if self.accelerator.is_main_process:
                self.logger.info("IQA metrics initialized successfully")
            return metrics
        except Exception as e:
            if self.accelerator.is_main_process:
                self.logger.warning(f"IQA metrics initialization failed: {e}")
            return None
            
    def get_scene_type(self, idx):
        """Determine scene type from sample index."""
        for scene_type, (start, end) in self.scene_ranges.items():
            if start <= idx <= end:
                return scene_type
        return 'reg'
    
    def evaluate(self):
        """Run IQA evaluation."""
        if not self.iqa_metrics:
            self.logger.warning("IQA metrics not available, skipping evaluation")
            return
            
        self.model.eval()
        torch.cuda.empty_cache()
        
        # Create save directories
        if self.accelerator.is_main_process:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_dir = os.path.join(self.manager.params.model_dir, f'gif_results_{timestamp}')
            png_dir = os.path.join(self.manager.params.model_dir, f'warped_results_{timestamp}')
            os.makedirs(gif_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)
        
        # Initialize metric containers
        iqa_dict = {scene: {} for scene in self.scene_ranges.keys()}
        for scene in iqa_dict:
            for metric in ['PSNR', 'SSIM', 'LPIPS']:
                for pair in ['12', '1ours', '1homogt', '1gtflow']:
                    iqa_dict[scene][f'{metric}_{pair}'] = []
        
        # Get stored results from GHOF evaluator
        ghof_evaluator = next(e for e in self.manager.evaluators if isinstance(e, GHOFEvaluator))
        if not ghof_evaluator or not ghof_evaluator.outputs:
            self.logger.error("No GHOF results available for IQA evaluation")
            return
            
        with torch.no_grad():
            for idx, (data_batch, output) in enumerate(ghof_evaluator.outputs):
                # Skip if no images available
                if "imgs_full" not in data_batch:
                    continue
                    
                # Process batch
                self._process_batch(data_batch, output, idx, iqa_dict)
        
        # Compute and print metrics
        self._compute_and_print_metrics(iqa_dict)
    
    def _process_batch(self, data_batch, output, batch_idx, iqa_dict):
        """Process a single batch for IQA metrics."""
        imgs_full = data_batch["imgs_full"]
        mask = data_batch["mask"]
        b, c, h, w = imgs_full.shape
        
        # Extract images
        img1 = imgs_full[:, :3]
        img2 = imgs_full[:, 3:]
        
        # Get flows
        flow_fw = output["flow_f"].permute(0, 3, 1, 2)
        flow_bw = output["flow_b"].permute(0, 3, 1, 2)
        homo_gt = data_batch["homo_field"]
        gt_flow = data_batch["gt_flow"]
        
        # Warp images and masks
        img2_warped = get_warp_flow(img2, flow_fw)
        img1_warped = get_warp_flow(img1, flow_bw)
        
        # Warp masks to track valid regions
        ones = torch.ones_like(img2)
        ours_mask = get_warp_flow(ones, flow_fw)
        homogt_mask = get_warp_flow(ones, homo_gt)
        gtflow_mask = get_warp_flow(ones, gt_flow)
        
        # Process each sample in batch
        for i in range(b):
            scene_type = self.get_scene_type(batch_idx * b + i)
            
            # Convert images to numpy
            img1_np = img1[i].detach().cpu().numpy().transpose(1, 2, 0)
            img2_np = img2[i].detach().cpu().numpy().transpose(1, 2, 0)
            img2_warped_np = img2_warped[i].detach().cpu().numpy().transpose(1, 2, 0)
            
            # Get masks
            mask_np = mask[i].detach().cpu().numpy().squeeze()
            ours_mask_np = ours_mask[i].detach().cpu().numpy()[0]  # Take first channel
            homogt_mask_np = homogt_mask[i].detach().cpu().numpy()[0]
            
            # Combine masks with strict threshold
            mask_threshold = 0.99  # Consider >0.99 as valid
            combined_mask = (mask_np > 0.5) & \
                          (ours_mask_np > mask_threshold) & \
                          (homogt_mask_np > mask_threshold)
            
            # Convert to float32 for IQA computation
            final_mask = combined_mask.astype(np.float32)
            
            # Ensure uint8 range
            img1_np = np.clip(img1_np, 0, 255).astype(np.uint8)
            img2_np = np.clip(img2_np, 0, 255).astype(np.uint8)
            img2_warped_np = np.clip(img2_warped_np, 0, 255).astype(np.uint8)
            
            # Compute IQA metrics with combined mask
            metrics_12 = compute_masked_metrics(img1_np, img2_np, final_mask, self.iqa_metrics)
            metrics_1ours = compute_masked_metrics(img1_np, img2_warped_np, final_mask, self.iqa_metrics)
            
            # Store metrics
            for metric_name in ['PSNR', 'SSIM', 'LPIPS']:
                iqa_dict[scene_type][f'{metric_name}_12'].append(metrics_12[metric_name])
                iqa_dict[scene_type][f'{metric_name}_1ours'].append(metrics_1ours[metric_name])
                
            # Print debug info for first sample
            if i == 0 and self.accelerator.is_main_process:
                valid_pixels = np.sum(final_mask > 0)
                total_pixels = final_mask.size
                self.logger.info(
                    f"Batch {batch_idx}, scene {scene_type}: "
                    f"Valid pixels: {valid_pixels}/{total_pixels} "
                    f"PSNR: {metrics_1ours['PSNR']:.4f}, "
                    f"SSIM: {metrics_1ours['SSIM']:.4f}, "
                    f"LPIPS: {metrics_1ours['LPIPS']:.4f}"
                )
    
    def _compute_and_print_metrics(self, iqa_dict):
        """Compute and print IQA metrics."""
        if not self.accelerator.is_main_process:
            return
            
        self.logger.info("\n=== IQA Metrics Summary ===")
        
        # Print per-scene metrics
        for scene in self.scene_ranges.keys():
            self.logger.info(f"\n{scene.upper()} Scene:")
            for pair in ['12', '1ours']:
                avg_psnr = np.mean(iqa_dict[scene][f'PSNR_{pair}']) if iqa_dict[scene][f'PSNR_{pair}'] else 0
                avg_ssim = np.mean(iqa_dict[scene][f'SSIM_{pair}']) if iqa_dict[scene][f'SSIM_{pair}'] else 0
                avg_lpips = np.mean(iqa_dict[scene][f'LPIPS_{pair}']) if iqa_dict[scene][f'LPIPS_{pair}'] else 0
                
                self.logger.info(
                    f"  [{pair}] "
                    f"PSNR: {avg_psnr:.4f}, "
                    f"SSIM: {avg_ssim:.4f}, "
                    f"LPIPS: {avg_lpips:.4f}"
                )
        
        # Print overall averages
        self.logger.info(f"\n🌍 OVERALL AVERAGE (5 scenes):")
        for pair in ['12', '1ours']:
            all_psnr = []
            all_ssim = []
            all_lpips = []
            for scene in self.scene_ranges.keys():
                all_psnr.extend(iqa_dict[scene][f'PSNR_{pair}'])
                all_ssim.extend(iqa_dict[scene][f'SSIM_{pair}'])
                all_lpips.extend(iqa_dict[scene][f'LPIPS_{pair}'])
            
            avg_psnr = np.mean(all_psnr) if all_psnr else 0
            avg_ssim = np.mean(all_ssim) if all_ssim else 0
            avg_lpips = np.mean(all_lpips) if all_lpips else 0
            
            self.logger.info(
                f"  [{pair}] "
                f"PSNR: {avg_psnr:.4f}, "
                f"SSIM: {avg_ssim:.4f}, "
                f"LPIPS: {avg_lpips:.4f}"
            )
        self.logger.info("==========================\n") 