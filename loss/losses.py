import math
import torch

import numpy as np
import torch.nn as nn
import mlflow

from kornia.geometry.linalg import transform_points


def init_mlflow_tracking(experiment_name,
                         run_name=None,
                         tracking_uri=None,
                         tags=None):
    """
    Initialize MLflow tracking for loss and metrics monitoring.
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Optional name for the specific run
        tracking_uri: Optional MLflow tracking server URI
        tags: Optional dictionary of tags for the run
    
    Returns:
        MLflow run object
    """
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        run = mlflow.start_run(run_name=run_name)

        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        print(f"MLflow tracking initialized. Experiment: {experiment_name}")
        if run_name:
            print(f"Run name: {run_name}")
        print(f"Run ID: {run.info.run_id}")

        return run

    except Exception as e:
        print(f"Warning: MLflow initialization failed: {e}")
        return None


class LossL1(nn.Module):

    def __init__(self, reduction='mean'):
        super(LossL1, self).__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def __call__(self, input, target):
        return self.loss(input, target)


class LossL2(nn.Module):

    def __init__(self):
        super(LossL2, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, input, target):
        return self.loss(input, target)


class Mask_Loss(nn.Module):

    def __init__(self, weight=(1, 1)):
        super(Mask_Loss, self).__init__()
        self.weight = weight
        self.loss = nn.BCELoss()

    def gen_weight(self, h, w):
        interval = self.weight[1] - self.weight[0]
        weight = interval * torch.arange(h) / h + self.weight[0]
        weight = torch.repeat_interleave(weight, w)
        return weight.view(1, 1, h, w)

    def __call__(self, x):
        bs, _, h, w = x.size()
        weight = self.gen_weight(h, w)
        weight = weight.repeat(bs, 1, 1, 1).to(x.device)
        mask_loss = self.loss(x, weight)
        return mask_loss


class NLLLaplace:
    """ Computes Negative Log Likelihood loss for a (single) Laplace distribution. """

    def __init__(self, reduction='mean', ratio=1.0):
        """
        Args:
            reduction: str, type of reduction to apply to loss
            ratio:
        """
        super().__init__()
        self.reduction = reduction
        self.ratio = ratio

    def __call__(self, gt_flow, est_flow, log_var, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var: estimated log variance, shape (b, 1, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        loss1 = math.sqrt(2) * torch.exp(-0.5 * log_var) * \
            torch.abs(gt_flow - est_flow)
        # each dimension is multiplied
        loss2 = 0.5 * log_var
        loss = loss1 + loss2

        if mask is not None:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(
                loss.detach()) & mask
        else:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(loss.detach())

        if torch.isnan(loss.detach()).sum().ge(1) or torch.isinf(
                loss.detach()).sum().ge(1):
            print('mask or inf in the loss ! ')

        if self.reduction == 'mean':
            if mask is not None:
                loss = torch.masked_select(loss, mask).mean()
            else:
                loss = loss.mean()
            return loss
        elif 'weighted_sum' in self.reduction:
            if mask is not None:
                loss = loss * mask.float()
                L = 0
                for bb in range(0, b):
                    norm_const = float(h) * float(w) / \
                        (mask[bb, ...].sum().float() + 1e-6)
                    L += loss[bb][mask[bb, ...] != 0].sum() * norm_const
                if 'normalized' in self.reduction:
                    return L / b
                return L

            if 'normalized' in self.reduction:
                return loss.sum() / b
            return loss
        else:
            raise ValueError


def triplet_loss(a, p, n, margin=1.0, exp=1, reduce=False, size_average=False):
    triplet_loss = nn.TripletMarginLoss(margin=margin,
                                        p=exp,
                                        reduce=reduce,
                                        size_average=size_average)
    return triplet_loss(a, p, n)


def _log_losses_to_mlflow(loss_dict, params, step=None):
    """
    Log loss components to MLflow for monitoring and visualization.
    
    Args:
        loss_dict: Dictionary containing loss values
        params: Parameters object containing experiment configuration
        step: Optional step number for logging
    """
    try:
        # Only log if MLflow is active
        if mlflow.active_run() is not None:
            for loss_name, loss_value in loss_dict.items():
                # Convert tensor to scalar if needed
                if torch.is_tensor(loss_value):
                    loss_value = loss_value.item()

                # Log with experiment context
                mlflow.log_metric(key=f"loss/{loss_name}",
                                  value=loss_value,
                                  step=step)

                # Also log stage-specific losses for better organization
                if hasattr(params, 'net_stg'):
                    mlflow.log_metric(
                        key=f"loss/stage_{params.net_stg}/{loss_name}",
                        value=loss_value,
                        step=step)

            # Log loss ratios for dynamic balancing analysis
            if 'supervise' in loss_dict and 'unsup' in loss_dict:
                sup_loss = loss_dict['supervise'].item() if torch.is_tensor(
                    loss_dict['supervise']) else loss_dict['supervise']
                unsup_loss = loss_dict['unsup'].item() if torch.is_tensor(
                    loss_dict['unsup']) else loss_dict['unsup']

                if unsup_loss > 0:
                    ratio = sup_loss / unsup_loss
                    mlflow.log_metric(key="loss/supervise_unsup_ratio",
                                      value=ratio,
                                      step=step)

    except Exception as e:
        # Silently handle MLflow logging errors to avoid disrupting training
        print(f"Warning: MLflow logging failed: {e}")
        pass


def compute_losses(data, endpoints, params):
    loss = {}

    # if params.net_type == 'DMHomo' and params.net_stg == 2:
    #     flow_b_gt, flow_f_gt = data["flow_gt_patch"][:,
    #                                                  :2, :, :], data["flow_gt_patch"][:, 2:, :, :]
    #     flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
    #     mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]
    #     fil_features = endpoints["fil_features"]

    #     # Loss Definition
    #     pl_criterion = LossL1(reduction='mean')
    #     mask_loss = Mask_Loss()

    #     loss['fil'] = params.fil_weight * (pl_criterion(fil_features["img1_patch_fea_warp"], fil_features["fea1_patch_warp"]) + pl_criterion(fil_features["img2_patch_fea_warp"],
    #                                                                                                                                          fil_features["fea2_patch_warp"]))

    #     # loss["triplet"] = triplet_loss(
    #     #     fil_features["img1_patch_fea"], fil_features["img2_patch_fea_warp"], fil_features["img2_patch_fea"]).mean() + triplet_loss(
    #     #         fil_features["img2_patch_fea"], fil_features["img1_patch_fea_warp"], fil_features["img1_patch_fea"]).mean()
    #     loss["unsup"] = params.unsup_weight * (pl_criterion(mask_f * fil_features["img1_patch_fea"], mask_f * fil_features["img2_patch_fea_warp"]) + pl_criterion(
    #         mask_b * fil_features["img2_patch_fea"], mask_b * fil_features["img1_patch_fea_warp"]))

    #     loss['mask_reg'] = params.mask_loss_weight * \
    #         (mask_loss(mask_b) + mask_loss(mask_f))
    #     loss['supervise'] = pl_criterion(
    #         mask_b * flow_b, mask_b * flow_b_gt) + pl_criterion(mask_f * flow_f, mask_f * flow_f_gt)
    #     # loss['supervise'] = pl_criterion(flow_b, flow_b_gt) + pl_criterion(flow_f, flow_f_gt)
    #     loss['total'] = loss['supervise'] + \
    #         loss['mask_reg'] + loss["unsup"] + loss['fil']

    if params.net_type == 'CamFlow' and params.net_stg == 0:
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
        mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]

        # Loss Definition
        pl_criterion = LossL1(reduction='mean')
        mask_loss = Mask_Loss()
        nll_laplace = NLLLaplace(reduction='mean')

        # loss["unsup"] = params.unsup_weight * (pl_criterion(mask_f * fil_features["img1_patch_fea"], mask_f * fil_features["img2_patch_fea_warp"]) + pl_criterion(
        #     mask_b * fil_features["img2_patch_fea"], mask_b * fil_features["img1_patch_fea_warp"]))

        loss['supervise'] = (
            nll_laplace(flow_f, flow_f_gt, torch.zeros_like(1 - mask_f)) +
            nll_laplace(flow_b, flow_b_gt, torch.zeros_like(1 - mask_b)))

        # loss['mask_reg'] = params.mask_loss_weight * \
        #     (mask_loss(mask_b) + mask_loss(mask_f))
        # loss['supervise'] = pl_criterion(
        #     mask_b * flow_b, mask_b * flow_b_gt) + pl_criterion(mask_f * flow_f, mask_f * flow_f_gt)
        # loss['supervise'] = pl_criterion(flow_b, flow_b_gt) + pl_criterion(flow_f, flow_f_gt)
        loss['total'] = loss['supervise']

    elif params.net_type == 'CamFlow' and params.net_stg == 1:
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
        mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]

        # Loss Definition
        pl_criterion = LossL1(reduction='mean')
        mask_loss = Mask_Loss()
        nll_laplace = NLLLaplace(reduction='mean')
        fil_features = endpoints["fil_features"]

        log_var_f = -torch.log(mask_f + 1e-8)
        log_var_b = -torch.log(mask_b + 1e-8)

        # loss["unsup"] = params.unsup_weight * (pl_criterion(mask_f * fil_features["img1_patch_fea"], mask_f * fil_features["img2_patch_fea_warp"]) + pl_criterion(
        #     mask_b * fil_features["img2_patch_fea"], mask_b * fil_features["img1_patch_fea_warp"]))

        # loss['supervise'] = (nll_laplace(flow_f, flow_f_gt, (1 - mask_f)) + nll_laplace(flow_b, flow_b_gt, (1 - mask_b)))
        loss['supervise'] = (nll_laplace(flow_f, flow_f_gt, log_var_f) +
                             nll_laplace(flow_b, flow_b_gt, log_var_b))

        loss["unsup"] = nll_laplace(
            fil_features["img1_patch_fea"],
            fil_features["img2_patch_fea_warp"], log_var_f) + nll_laplace(
                fil_features["img2_patch_fea"],
                fil_features["img1_patch_fea_warp"], log_var_b)

        # loss['mask_reg'] = params.mask_loss_weight * \
        #     (mask_loss(mask_b) + mask_loss(mask_f))
        # loss['supervise'] = pl_criterion(
        #     mask_b * flow_b, mask_b * flow_b_gt) + pl_criterion(mask_f * flow_f, mask_f * flow_f_gt)
        # loss['supervise'] = pl_criterion(flow_b, flow_b_gt) + pl_criterion(flow_f, flow_f_gt)
        loss['total'] = loss['supervise']

    elif params.net_type == 'CamFlow' and params.net_stg == 1.1:
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
        mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]

        # Loss Definition
        mask_loss = Mask_Loss()
        pl_criterion = LossL1(reduction='mean')
        nll_laplace = NLLLaplace(reduction='mean')
        fil_features = endpoints["fil_features"]

        log_var_f = 1 - torch.sigmoid(mask_f)
        log_var_b = 1 - torch.sigmoid(mask_b)

        # loss["unsup"] = params.unsup_weight * (pl_criterion(mask_f * fil_features["img1_patch_fea"], mask_f * fil_features["img2_patch_fea_warp"]) + pl_criterion(
        #     mask_b * fil_features["img2_patch_fea"], mask_b * fil_features["img1_patch_fea_warp"]))

        # loss['supervise'] = (nll_laplace(flow_f, flow_f_gt, (1 - mask_f)) + nll_laplace(flow_b, flow_b_gt, (1 - mask_b)))
        loss['supervise'] = (nll_laplace(flow_f, flow_f_gt, log_var_f) +
                             nll_laplace(flow_b, flow_b_gt, log_var_b))

        loss["unsup"] = nll_laplace(
            fil_features["img1_patch_fea"],
            fil_features["img2_patch_fea_warp"], log_var_f) + nll_laplace(
                fil_features["img2_patch_fea"],
                fil_features["img1_patch_fea_warp"], log_var_b)

        # loss['mask_reg'] = params.mask_loss_weight * \
        #     (mask_loss(mask_b) + mask_loss(mask_f))
        # loss['supervise'] = pl_criterion(
        #     mask_b * flow_b, mask_b * flow_b_gt) + pl_criterion(mask_f * flow_f, mask_f * flow_f_gt)
        # loss['supervise'] = pl_criterion(flow_b, flow_b_gt) + pl_criterion(flow_f, flow_f_gt)
        loss['total'] = loss['supervise']

    elif params.net_type == 'CamFlow' and params.net_stg == 1.2:
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
        mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]

        # Loss Definition
        mask_loss = Mask_Loss()
        pl_criterion = LossL1(reduction='mean')
        nll_laplace = NLLLaplace(reduction='mean')
        fil_features = endpoints["fil_features"]

        log_var_f = 1 - torch.sigmoid(mask_f)
        log_var_b = 1 - torch.sigmoid(mask_b)

        # loss["unsup"] = params.unsup_weight * (pl_criterion(mask_f * fil_features["img1_patch_fea"], mask_f * fil_features["img2_patch_fea_warp"]) + pl_criterion(
        #     mask_b * fil_features["img2_patch_fea"], mask_b * fil_features["img1_patch_fea_warp"]))

        # loss['supervise'] = (nll_laplace(flow_f, flow_f_gt, (1 - mask_f)) + nll_laplace(flow_b, flow_b_gt, (1 - mask_b)))
        loss['supervise'] = (nll_laplace(flow_f, flow_f_gt, log_var_f) +
                             nll_laplace(flow_b, flow_b_gt, log_var_b))

        loss["unsup"] = nll_laplace(
            fil_features["img1_patch_fea"],
            fil_features["img2_patch_fea_warp"], log_var_f) + nll_laplace(
                fil_features["img2_patch_fea"],
                fil_features["img1_patch_fea_warp"], log_var_b)

        # Dynamic balancing: scale unsupervised loss by the ratio of supervised to unsupervised loss
        balance_ratio = loss["supervise"] / (loss["unsup"] + 1e-7)
        loss['total'] = loss['supervise'] * params.sup_weight + loss[
            "unsup"] * balance_ratio * params.unsup_weight

    elif params.net_type == 'CamFlow' and params.net_stg == 4:
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
        mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]
        fil_features = endpoints["fil_features"]

        # Loss Definition
        pl_criterion = LossL1(reduction='mean')
        # mask_loss = Mask_Loss()
        nll_laplace = NLLLaplace(reduction='mean')

        loss['supervise'] = (nll_laplace(flow_f, flow_f_gt, mask_f) +
                             nll_laplace(flow_b, flow_b_gt, mask_b))

        loss["unsup"] = nll_laplace(
            fil_features["img1_patch_fea"],
            fil_features["img2_patch_fea_warp"], mask_f) + nll_laplace(
                fil_features["img2_patch_fea"],
                fil_features["img1_patch_fea_warp"], mask_b)

        # Dynamic balancing: scale unsupervised loss by the ratio of supervised to unsupervised loss
        balance_ratio = loss["supervise"] / (loss["unsup"] + 1e-7)
        loss['total'] = loss['supervise'] * params.sup_weight + loss[
            "unsup"] * balance_ratio * params.unsup_weight

    elif params.net_type == 'CamFlow' and params.net_stg == 4.1:
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
        mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]
        fil_features = endpoints["fil_features"]

        # Loss Definition
        pl_criterion = LossL1(reduction='mean')
        # mask_loss = Mask_Loss()
        nll_laplace = NLLLaplace(reduction='mean')

        loss['supervise'] = (nll_laplace(flow_f, flow_f_gt, mask_f) +
                             nll_laplace(flow_b, flow_b_gt, mask_b))

        loss["unsup"] = nll_laplace(
            fil_features["img1_patch_fea"],
            fil_features["img2_patch_fea_warp"], mask_f) + nll_laplace(
                fil_features["img2_patch_fea"],
                fil_features["img1_patch_fea_warp"], mask_b)

        loss['total'] = loss["unsup"]

    elif params.net_type == 'CamFlow' and params.net_stg == 4.2:
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
        mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]
        fil_features = endpoints["fil_features"]

        # Loss Definition
        pl_criterion = LossL1(reduction='mean')
        nll_laplace = NLLLaplace(reduction='mean')

        loss['supervise'] = (nll_laplace(flow_f, flow_f_gt, mask_f) +
                             nll_laplace(flow_b, flow_b_gt, mask_b))

        loss["unsup"] = nll_laplace(
            endpoints["img1_patch"],
            endpoints["img2_patch_warp"], mask_f) + nll_laplace(
                endpoints["img2_patch"], endpoints["img1_patch_warp"], mask_b)

        loss['total'] = loss["unsup"]

    elif params.net_type == 'CamFlow' and params.net_stg == 4.3:
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
        mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]
        fil_features = endpoints["fil_features"]

        # Loss Definition
        pl_criterion = LossL1(reduction='mean')
        # mask_loss = Mask_Loss()
        nll_laplace = NLLLaplace(reduction='mean')

        loss['supervise'] = (nll_laplace(flow_f, flow_f_gt, mask_f) +
                             nll_laplace(flow_b, flow_b_gt, mask_b))

        loss["unsup"] = nll_laplace(
            fil_features["img1_patch_fea"],
            fil_features["img2_patch_fea_warp"], mask_f) + nll_laplace(
                fil_features["img2_patch_fea"],
                fil_features["img1_patch_fea_warp"], mask_b)

        loss['total'] = loss["unsup"]

    elif params.net_type == 'CamFlow' and params.net_stg == 4.4:
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
        mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]
        fil_features = endpoints["fil_features"]

        # Loss Definition
        pl_criterion = LossL1(reduction='mean')
        # mask_loss = Mask_Loss()
        nll_laplace = NLLLaplace(reduction='mean')

        log_var_f = 1 - torch.sigmoid(mask_f)
        log_var_b = 1 - torch.sigmoid(mask_b)

        loss['supervise'] = (nll_laplace(flow_f, flow_f_gt, log_var_f) +
                             nll_laplace(flow_b, flow_b_gt, log_var_b))

        loss["unsup"] = nll_laplace(
            fil_features["img1_patch_fea"],
            fil_features["img2_patch_fea_warp"], log_var_f) + nll_laplace(
                fil_features["img2_patch_fea"],
                fil_features["img1_patch_fea_warp"], log_var_b)

        # Dynamic balancing: scale unsupervised loss by the ratio of supervised to unsupervised loss
        balance_ratio = loss["supervise"] / (loss["unsup"] + 1e-7)
        loss['total'] = loss['supervise'] * params.sup_weight + loss[
            "unsup"] * balance_ratio * params.unsup_weight

    elif params.net_type == 'CamFlow' and params.net_stg == 2:
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
        mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]

        # Loss Definition
        pl_criterion = LossL1(reduction='mean')
        mask_loss = Mask_Loss()
        nll_laplace = NLLLaplace(reduction='mean')

        mask_var_scale = params.mask_var_scale

        # loss["unsup"] = params.unsup_weight * (pl_criterion(mask_f * fil_features["img1_patch_fea"], mask_f * fil_features["img2_patch_fea_warp"]) + pl_criterion(
        #     mask_b * fil_features["img2_patch_fea"], mask_b * fil_features["img1_patch_fea_warp"]))

        loss['supervise'] = (nll_laplace(flow_f, flow_f_gt,
                                         (1 - mask_f) * mask_var_scale) /
                             +nll_laplace(flow_b, flow_b_gt,
                                          (1 - mask_b) * mask_var_scale))

        # loss['mask_reg'] = params.mask_loss_weight * \
        #     (mask_loss(mask_b) + mask_loss(mask_f))
        # loss['supervise'] = pl_criterion(
        #     mask_b * flow_b, mask_b * flow_b_gt) + pl_criterion(mask_f * flow_f, mask_f * flow_f_gt)
        # loss['supervise'] = pl_criterion(flow_b, flow_b_gt) + pl_criterion(flow_f, flow_f_gt)
        loss['total'] = loss['supervise']

    # TODO: 加入photoloss
    elif params.net_type == 'CamFlow' and params.net_stg == 3:
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
        mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]
        fil_features = endpoints["fil_features"]

        # Loss Definition
        pl_criterion = LossL1(reduction='mean')
        mask_loss = Mask_Loss()
        nll_laplace = NLLLaplace(reduction='mean')

        # mask_var_scale = params.mask_var_scale

        # loss["unsup"] = params.unsup_weight * (pl_criterion(mask_f * fil_features["img1_patch_fea"], mask_f * fil_features["img2_patch_fea_warp"]) + pl_criterion(
        #     mask_b * fil_features["img2_patch_fea"], mask_b * fil_features["img1_patch_fea_warp"]))
        loss["unsup"] = nll_laplace(
            fil_features["img1_patch_fea"],
            fil_features["img2_patch_fea_warp"], mask_f) + nll_laplace(
                fil_features["img2_patch_fea"],
                fil_features["img1_patch_fea_warp"], mask_b)

        loss['supervise'] = nll_laplace(
            flow_f, flow_f_gt, mask_f) + nll_laplace(flow_b, flow_b_gt, mask_b)

        # loss['mask_reg'] = params.mask_loss_weight * \
        #     (mask_loss(mask_b) + mask_loss(mask_f))
        # loss['supervise'] = pl_criterion(
        #     mask_b * flow_b, mask_b * flow_b_gt) + pl_criterion(mask_f * flow_f, mask_f * flow_f_gt)
        # loss['supervise'] = pl_criterion(flow_b, flow_b_gt) + pl_criterion(flow_f, flow_f_gt)
        loss['total'] = loss['supervise'] + params.unsup_weight * loss["unsup"]

    # TODO: 加入photoloss
    elif params.net_type == 'CamFlow' and params.net_stg == 13:
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
        mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]
        fil_features = endpoints["fil_features"]

        # Loss Definition
        pl_criterion = LossL1(reduction='mean')
        mask_loss = Mask_Loss()
        nll_laplace = NLLLaplace(reduction='mean')

        # mask_var_scale = params.mask_var_scale

        # loss["unsup"] = params.unsup_weight * (pl_criterion(mask_f * fil_features["img1_patch_fea"], mask_f * fil_features["img2_patch_fea_warp"]) + pl_criterion(
        #     mask_b * fil_features["img2_patch_fea"], mask_b * fil_features["img1_patch_fea_warp"]))
        loss["unsup"] = nll_laplace(
            fil_features["img1_patch_fea"],
            fil_features["img2_patch_fea_warp"], mask_f) + nll_laplace(
                fil_features["img2_patch_fea"],
                fil_features["img1_patch_fea_warp"], mask_b)

        loss['supervise'] = nll_laplace(
            flow_f, flow_f_gt, mask_f) + nll_laplace(flow_b, flow_b_gt, mask_b)

        # loss['mask_reg'] = params.mask_loss_weight * \
        #     (mask_loss(mask_b) + mask_loss(mask_f))
        # loss['supervise'] = pl_criterion(
        #     mask_b * flow_b, mask_b * flow_b_gt) + pl_criterion(mask_f * flow_f, mask_f * flow_f_gt)
        # loss['supervise'] = pl_criterion(flow_b, flow_b_gt) + pl_criterion(flow_f, flow_f_gt)
        # Dynamic balancing with weight normalization
        balance_ratio = loss["supervise"] / (loss["unsup"] + 1e-7)
        loss['total'] = loss['supervise'] / params.unsup_weight + loss[
            "unsup"] * balance_ratio

    # TODO: 加入photoloss
    elif params.net_type == 'CamFlow' and params.net_stg == 23:
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
        mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]
        fil_features = endpoints["fil_features"]

        # Loss Definition
        pl_criterion = LossL1(reduction='mean')
        mask_loss = Mask_Loss()
        nll_laplace = NLLLaplace(reduction='mean')

        # mask_var_scale = params.mask_var_scale

        # loss["unsup"] = params.unsup_weight * (pl_criterion(mask_f * fil_features["img1_patch_fea"], mask_f * fil_features["img2_patch_fea_warp"]) + pl_criterion(
        #     mask_b * fil_features["img2_patch_fea"], mask_b * fil_features["img1_patch_fea_warp"]))
        loss["unsup"] = nll_laplace(
            fil_features["img1_patch_fea"],
            fil_features["img2_patch_fea_warp"], mask_f) + nll_laplace(
                fil_features["img2_patch_fea"],
                fil_features["img1_patch_fea_warp"], mask_b)

        # loss['mask_reg'] = params.mask_loss_weight * \
        #     (mask_loss(mask_b) + mask_loss(mask_f))
        # loss['supervise'] = pl_criterion(
        #     mask_b * flow_b, mask_b * flow_b_gt) + pl_criterion(mask_f * flow_f, mask_f * flow_f_gt)
        # loss['supervise'] = pl_criterion(flow_b, flow_b_gt) + pl_criterion(flow_f, flow_f_gt)
        loss['total'] = loss["unsup"]

        # print(f"gpu:{torch.cuda.current_device()} loss:{loss['total']}")

    elif params.net_type == 'HomoGAN':
        flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
            "flow_gt_patch"][:, 2:, :, :]
        flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]

        # Loss Definition
        pl_criterion = LossL1(reduction='mean')

        loss['supervise'] = pl_criterion(flow_b, flow_b_gt) + pl_criterion(
            flow_f, flow_f_gt)
        loss['total'] = loss['supervise']

    # MLflow logging for loss monitoring
    _log_losses_to_mlflow(loss, params)
    return loss


def compute_metrics(data, output, manager):
    metrics = {}
    with torch.no_grad():
        # compute metrics
        B = data["label"].size()[0]
        outputs = np.argmax(output["p"].detach().cpu().numpy(), axis=1)
        accuracy = np.sum(
            outputs.astype(np.int32) ==
            data["label"].detach().cpu().numpy().astype(np.int32)) / B
        metrics['accuracy'] = accuracy

        # MLflow logging for metrics monitoring
        try:
            if mlflow.active_run() is not None:
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"metrics/{metric_name}", metric_value)
        except Exception as e:
            print(f"Warning: MLflow metrics logging failed: {e}")

        return metrics


# def ComputeErrH(src, dst, H):
#     src = src.unsqueeze(0).unsqueeze(0).detach().cpu().numpy().astype('float64')
#     dst = dst.detach().cpu().numpy().astype('float64')
#     src_warp = cv2.perspectiveTransform(src, H.squeeze(0).detach().cpu().numpy()).reshape(2)
#     err = np.sqrt((dst[0] - src_warp[0])**2 + (dst[1] - src_warp[1])**2)
#     return err


def ComputeErrH_kornia(src, dst, H):
    # src,dst:(B, N, 2)
    # H:(N, 3, 3)
    src_warp = transform_points(H, src)
    err = torch.sqrt((src_warp[:, :, 0] - src[:, :, 0])**2 +
                     (src_warp[:, :, 1] - dst[:, :, 1])**2)
    return err


def ComputeErrH_v2(src, dst, H):
    '''
    :param src: B, N, 2
    :param dst: B, N, 2
    :param H: B, 3, 3
    '''
    src, dst = src.unsqueeze(0).unsqueeze(0), dst.unsqueeze(0).unsqueeze(0)
    src_warp = transform_points(H.unsqueeze(0), src)
    err = torch.linalg.norm(dst - src_warp)
    return err


def ComputeErrFlow(src, dst, flow):
    src_t = src + flow[int(src[1]), int(src[0])]
    error = torch.linalg.norm(dst - src_t)
    return error


def ComputeErr(src, dst):
    error = torch.linalg.norm(dst - src)
    return error


# def compute_eval_results(data_batch, output_batch, accelerator):
#     imgs_full = data_batch["imgs_gray_full"]
#     accelerator.print(f"imgs_full shape {imgs_full.shape}")

#     # pt_set = list(map(eval, data_batch["pt_set"]))
#     pt_set = data_batch["pt_set"]
#     accelerator.print(f"pt_set shape {pt_set.shape}")
#     # pt_set = list(map(lambda x: x['matche_pts'], pt_set))

#     batch_size, _, img_h, img_w = imgs_full.shape
#     Homo_b = output_batch["Homo_b"]
#     Homo_f = output_batch["Homo_f"]
#     accelerator.print(f"Homo_b shape {Homo_b.shape}")

#     src = pt_set[:, :6, 0]
#     dst = pt_set[:, :6, 1]

#     # errs_m = []
#     # pred_errs = []
#     # for i in range(batch_size):
#     #     # pts = torch.Tensor(pt_set[i]).to(accelerator.device)
#     #     pts = pt_set[i]
#     #     err = 0
#     #     for j in range(6):
#     #         p1 = pts[j][0]
#     #         p2 = pts[j][1]
#     #         src, dst = p1, p2
#     #         pred_err = min(ComputeErrH(src=src, dst=dst, H=Homo_b[i]), ComputeErrH(src=dst, dst=src, H=Homo_f[i]))
#     #         err += pred_err
#     #         pred_errs.append(pred_err)
#     #     err /= 6
#     #     errs_m.append(err)

#     # errs_b,errs_b:(B, 1)
#     errs_b = torch.mean(ComputeErrH_kornia(src, dst, Homo_b), 1)
#     errs_f = torch.mean(ComputeErrH_kornia(dst, src, Homo_f), 1)

#     for i, _ in enumerate(errs_b):
#         errs_b[i] = torch.minimum(errs_b[i], errs_f[i])
#     # eval_results = {"errors_m": np.array(errs_m)}
#     accelerator.print(f"eval_results {errs_b}")
#     return errs_b


def compute_eval_results(data_batch, output_batch):
    imgs_full = data_batch["imgs_gray_full"]
    device = imgs_full.device
    batch_size, _, img_h, img_w = imgs_full.shape

    pt_set = data_batch["pt_set"]
    flow_f = output_batch["flow_f"]
    flow_b = output_batch["flow_b"]
    # # [TODO] Test I33
    # flow_f = torch.zeros_like(flow_f)
    # flow_b = torch.zeros_like(flow_b)
    # print(f"flow_f shape {flow_f.shape}")

    # for unit test
    # for i in range(flow_f.shape[0]):
    #     flow_f[i] = torch.zeros_like(flow_f[i])
    #     flow_b[i] = torch.zeros_like(flow_b[i])

    errs_m = []
    pred_errs = []
    for i in range(batch_size):
        pts = torch.Tensor(pt_set[i]).to(device)
        err = 0
        for j in range(6):
            p1 = pts[j][0]
            p2 = pts[j][1]
            src, dst = p1, p2
            pred_err = min(ComputeErrFlow(src=src, dst=dst, flow=flow_f[i]),
                           ComputeErrFlow(src=dst, dst=src, flow=flow_b[i]))
            err += pred_err
            pred_errs.append(pred_err)
        err /= 6

        errs_m.append(err)

    return errs_m
