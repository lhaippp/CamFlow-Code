import torch
import numpy as np
import pyiqa


def compute_masked_metrics(img1_rgb, img2_rgb, mask, iqa_metrics=None):
    """
    计算mask区域内两个图像之间的图像质量指标
    Args:
        img1_rgb: 第一个图像，numpy数组，形状为[H, W, 3]，uint8类型，RGB格式
        img2_rgb: 第二个图像，numpy数组，形状为[H, W, 3]，uint8类型，RGB格式
        mask: mask数组，形状为[H, W]，float类型，0-1值
        iqa_metrics: 预创建的IQA指标字典，如果None则会创建新的
    Returns:
        dict: 包含PSNR、SSIM、LPIPS指标的字典
    """
    # 调试信息（只在需要时打印）
    # print(f"compute_masked_metrics input shapes: img1 {img1_rgb.shape}, img2 {img2_rgb.shape}, mask {mask.shape}")

    # 检查输入形状
    if len(img1_rgb.shape) != 3 or img1_rgb.shape[2] != 3:
        raise ValueError(f"img1_rgb should be [H, W, 3], got {img1_rgb.shape}")
    if len(img2_rgb.shape) != 3 or img2_rgb.shape[2] != 3:
        raise ValueError(f"img2_rgb should be [H, W, 3], got {img2_rgb.shape}")
    if len(mask.shape) != 2:
        raise ValueError(f"mask should be [H, W], got {mask.shape}")

    # 确保所有数组形状匹配
    h1, w1, c1 = img1_rgb.shape
    h2, w2, c2 = img2_rgb.shape
    hm, wm = mask.shape

    if h1 != h2 or w1 != w2 or h1 != hm or w1 != wm:
        raise ValueError(
            f"Shape mismatch: img1 {img1_rgb.shape}, img2 {img2_rgb.shape}, mask {mask.shape}"
        )

    # 将mask扩展到三个通道
    mask_3ch = np.stack([mask, mask, mask], axis=2)  # [H, W, 3]

    # 应用mask到图像上
    img1_masked = (img1_rgb.astype(np.float32) * mask_3ch).astype(np.uint8)
    img2_masked = (img2_rgb.astype(np.float32) * mask_3ch).astype(np.uint8)

    # 转换为float32, 0-1范围
    img1_float = img1_masked.astype(np.float32) / 255.0
    img2_float = img2_masked.astype(np.float32) / 255.0

    # 转换为torch tensor (H,W,C) -> (1,C,H,W)
    img1_tensor = torch.from_numpy(img1_float).permute(2, 0, 1).unsqueeze(0)
    img2_tensor = torch.from_numpy(img2_float).permute(2, 0, 1).unsqueeze(0)

    # 使用传入的指标或创建新的
    if iqa_metrics is None:
        metrics = {
            'PSNR': pyiqa.create_metric('psnr'),
            'SSIM': pyiqa.create_metric('ssim'),
            'LPIPS': pyiqa.create_metric('lpips'),
        }
    else:
        metrics = iqa_metrics

    results = {}
    with torch.no_grad():
        for metric_name, metric in metrics.items():
            try:
                score = metric(img2_tensor, img1_tensor).item()
                results[metric_name] = score
            except Exception as e:
                print(f"Error computing {metric_name}: {e}")
                results[metric_name] = float('nan')

    return results


def compute_iqa_metrics_batch(data_batch, output_batch):
    """
    计算一个batch的IQA指标
    Args:
        data_batch: 数据批次，包含imgs_full和mask
        output_batch: 模型输出，包含flow_f
    Returns:
        dict: 包含各类别IQA指标的字典
    """
    from model.utils import get_warp_flow

    # 获取数据
    video_names = data_batch["video_names"]
    imgs_full = data_batch["imgs_full"]  # [B, 6, H, W]
    homo_gt = data_batch["homo_field"]  # [B, 2, H, W]
    mask = data_batch["mask"]  # [B, H, W]
    flow_fw = output_batch["flow_f"]  # [B, H, W, 2]

    # 初始化结果字典
    results = {
        'RE': {
            'img1_vs_img2': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            },
            'img1_vs_ours': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            },
            'img1_vs_gt': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            }
        },
        'Fog': {
            'img1_vs_img2': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            },
            'img1_vs_ours': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            },
            'img1_vs_gt': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            }
        },
        'Dark': {
            'img1_vs_img2': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            },
            'img1_vs_ours': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            },
            'img1_vs_gt': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            }
        },
        'Rain': {
            'img1_vs_img2': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            },
            'img1_vs_ours': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            },
            'img1_vs_gt': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            }
        },
        'SNOW': {
            'img1_vs_img2': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            },
            'img1_vs_ours': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            },
            'img1_vs_gt': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            }
        }
    }

    b, c, h, w = imgs_full.shape

    # 提取img1和img2
    img1 = imgs_full[:, :3, :, :]  # [B, 3, H, W]
    img2 = imgs_full[:, 3:, :, :]  # [B, 3, H, W]

    # 转换flow格式用于扭曲
    flow_fw_warp = flow_fw.permute(0, 3, 1, 2)  # [B, 2, H, W]

    # 使用网络流场扭曲img2
    img2_warped_ours = get_warp_flow(img2, flow_fw_warp)  # [B, 3, H, W]

    # 使用ground truth homography扭曲img2
    img2_warped_homogt = get_warp_flow(img2, homo_gt)  # [B, 3, H, W]

    # 处理每个样本
    for i, video_name in enumerate(video_names):
        if video_name not in results:
            continue

        # 转换为CPU numpy数组
        img1_single = img1[i].detach().cpu().numpy()  # [3, H, W]
        img2_single = img2[i].detach().cpu().numpy()  # [3, H, W]
        img2_warped_ours_single = img2_warped_ours[i].detach().cpu().numpy(
        )  # [3, H, W]
        img2_warped_homogt_single = img2_warped_homogt[i].detach().cpu().numpy(
        )  # [3, H, W]
        mask_single = mask[i].detach().cpu().numpy()  # [H, W]

        # Check shapes for debugging
        if i == 0:  # Only print for first sample to avoid spam
            print(
                f"IQA Debug - Sample shapes: img1 {img1_single.shape}, mask {mask_single.shape}"
            )

        # 转换为uint8格式
        img1_single = np.clip(img1_single, 0, 255).astype(np.uint8)
        img2_single = np.clip(img2_single, 0, 255).astype(np.uint8)
        img2_warped_ours_single = np.clip(img2_warped_ours_single, 0,
                                          255).astype(np.uint8)
        img2_warped_homogt_single = np.clip(img2_warped_homogt_single, 0,
                                            255).astype(np.uint8)

        # 确保所有图像都是正确的形状：[C, H, W] -> [H, W, C]
        if len(img1_single.shape) == 3 and img1_single.shape[0] == 3:
            img1_rgb = np.transpose(img1_single, (1, 2, 0))
        else:
            print(f"Warning: unexpected img1 shape {img1_single.shape}")
            continue

        if len(img2_single.shape) == 3 and img2_single.shape[0] == 3:
            img2_rgb = np.transpose(img2_single, (1, 2, 0))
        else:
            print(f"Warning: unexpected img2 shape {img2_single.shape}")
            continue

        if len(img2_warped_ours_single.shape
               ) == 3 and img2_warped_ours_single.shape[0] == 3:
            img2_warped_ours_rgb = np.transpose(img2_warped_ours_single,
                                                (1, 2, 0))
        else:
            print(
                f"Warning: unexpected img2_warped_ours shape {img2_warped_ours_single.shape}"
            )
            continue

        if len(img2_warped_homogt_single.shape
               ) == 3 and img2_warped_homogt_single.shape[0] == 3:
            img2_warped_homogt_rgb = np.transpose(img2_warped_homogt_single,
                                                  (1, 2, 0))
        else:
            print(
                f"Warning: unexpected img2_warped_homogt shape {img2_warped_homogt_single.shape}"
            )
            continue

        # 确保mask是2D的
        if len(mask_single.shape) > 2:
            mask_single = mask_single.squeeze()

        # Final shape check (optional debug)
        if i == 0:  # Only print for first sample
            print(
                f"IQA Debug - Final RGB shapes: {img1_rgb.shape}, mask: {mask_single.shape}"
            )

        # 计算IQA指标
        # img1 vs img2
        metrics_img1_vs_img2 = compute_masked_metrics(img1_rgb, img2_rgb,
                                                      mask_single)
        # img1 vs our warped image
        metrics_img1_vs_ours = compute_masked_metrics(img1_rgb,
                                                      img2_warped_ours_rgb,
                                                      mask_single)
        # img1 vs gt warped image
        metrics_img1_vs_gt = compute_masked_metrics(img1_rgb,
                                                    img2_warped_homogt_rgb,
                                                    mask_single)

        # 存储结果
        for metric_name in ['PSNR', 'SSIM', 'LPIPS']:
            results[video_name]['img1_vs_img2'][metric_name].append(
                metrics_img1_vs_img2[metric_name])
            results[video_name]['img1_vs_ours'][metric_name].append(
                metrics_img1_vs_ours[metric_name])
            results[video_name]['img1_vs_gt'][metric_name].append(
                metrics_img1_vs_gt[metric_name])

    return results


def print_iqa_summary(iqa_results):
    """打印IQA指标汇总"""
    categories = ['RE', 'Fog', 'Dark', 'Rain', 'SNOW']
    comparisons = ['img1_vs_img2', 'img1_vs_ours', 'img1_vs_gt']
    metrics = ['PSNR', 'SSIM', 'LPIPS']

    print("\n=== IQA Metrics Summary ===")
    for comparison in comparisons:
        print(f"\n--- {comparison} ---")
        for metric in metrics:
            print(f"{metric}:")
            values = []
            for cat in categories:
                cat_values = iqa_results[cat][comparison][metric]
                if cat_values:
                    avg_val = np.mean(cat_values)
                    print(
                        f"  {cat}: {avg_val:.4f} ({len(cat_values)} samples)")
                    values.extend(cat_values)
                else:
                    print(f"  {cat}: N/A (0 samples)")

            if values:
                overall_avg = np.mean(values)
                print(f"  Overall: {overall_avg:.4f} ({len(values)} samples)")
    print("===========================\n")
