from dust3r.losses import *
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchmetrics import JaccardIndex, Accuracy
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import lpips
from large_spatial_model.utils.gaussian_model import GaussianModel
from large_spatial_model.utils.cuda_splatting import render, DummyPipeline
from einops import rearrange
from large_spatial_model.utils.camera_utils import get_scaled_camera
from dust3r.inference import make_batch_symmetric

LABELS = ['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other']
NUM_LABELS = len(LABELS) + 1
PALETTE = plt.cm.get_cmap('tab10', NUM_LABELS)
COLORS_LIST = [PALETTE(i)[:3] for i in range(NUM_LABELS)]
COLORS = torch.tensor(COLORS_LIST, dtype=torch.float32)


def calculate_depth_metrics(pred_depth, gt_depth):
    # create mask
    mask = (gt_depth > 0) & (pred_depth > 0)

    # apply mask
    gt_depth_masked = gt_depth[mask]
    pred_depth_masked = pred_depth[mask]

    # avoid division by zero and handle empty masks
    if gt_depth_masked.numel() == 0 or pred_depth_masked.numel() == 0:
        nan = torch.tensor(float('nan'), device=pred_depth.device)
        return nan, nan

    median_gt = torch.median(gt_depth_masked)
    median_pred = torch.median(pred_depth_masked)

    if torch.isclose(median_pred, torch.tensor(0.0, device=pred_depth.device)):
        nan = torch.tensor(float('nan'), device=pred_depth.device)
        return nan, nan

    # calculate the scale
    scale = median_gt / median_pred

    # scale the pred depth
    pred_depth_masked = pred_depth_masked * scale

    # calculate the metrics
    rel_err = torch.abs(gt_depth_masked - pred_depth_masked) / gt_depth_masked

    # avoid NaN in relative error
    rel_err[torch.isnan(rel_err)] = 0

    # tau (mean of max(pred/gt, gt/pred) < 1.03)
    ratio = torch.max(pred_depth_masked / gt_depth_masked, gt_depth_masked / pred_depth_masked)

    tau = (ratio < 1.03).float().mean()

    return rel_err.mean() * 100, tau * 100


def _to_bchw(tensor):
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f'Unsupported tensor shape for BCHW conversion: {tensor.shape}')


def _to_bhw(tensor):
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 4:
        if tensor.shape[1] == 1:
            return tensor[:, 0]
        if tensor.shape[-1] == 1:
            return tensor[..., 0]
    raise ValueError(f'Unsupported tensor shape for BHW conversion: {tensor.shape}')

class KWRegr3D(Regr3D):
    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None, **kwargs):
        return super().get_all_pts3d(gt1, gt2, pred1, pred2, dist_clip)

class L2Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance

class L1Loss (LLoss):
    """ Manhattan distance between 3d points """

    def distance(self, a, b):
        return torch.abs(a - b).mean()  # L1 distance

L2 = L2Loss()
L1 = L1Loss()

def merge_and_split_predictions(pred1, pred2):
    merged = {}
    for key in ['scales', 'rotations', 'covs', 'opacities', 'sh_coeffs', 'means', 'gs_feats']:
        merged_pred = torch.stack([pred1[key], pred2[key]], dim=1)
        merged_pred = rearrange(merged_pred, 'b v h w ... -> b (v h w) ...')
        merged[key] = merged_pred

    # Split along the batch dimension
    batch_size = next(iter(merged.values())).shape[0]
    split = [{key: value[i] for key, value in merged.items()} for i in range(batch_size)]
    
    return split

class GaussianLoss(MultiLoss):
    def __init__(self, ssim_weight=0.2, feature_loss_weight=0.2, lables=['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other']):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.feature_loss_weight = feature_loss_weight
        self.labels = lables
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
        self.lpips_vgg = lpips.LPIPS(net='vgg').cuda()
        self.miou = JaccardIndex(num_classes=len(self.labels) + 1, task='multiclass', ignore_index=0)
        self.accuracy = Accuracy(num_classes=len(self.labels) + 1, task='multiclass', ignore_index=0)
        self.pipeline = DummyPipeline()
        # bg_color
        self.register_buffer('bg_color', torch.tensor([0.0, 0.0, 0.0]).cuda())
        
    def get_name(self):
        return f'GaussianLoss(ssim_weight={self.ssim_weight})'

    def compute_loss(self, gt1, gt2, pred1, pred2, target_view=None, target_views=None, model=None, **kwargs):
        # render images
        # 1. merge predictions
        pred = merge_and_split_predictions(pred1, pred2)
        
        # 2. calculate optimal scaling
        pred_pts1 = pred1['means']
        pred_pts2 = pred2['means']
        # convert to camera1 coordinates
        # everything is normalized w.r.t. camera of view1
        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'].to(in_camera1.device))  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt2['pts3d'].to(in_camera1.device))  # B,H,W,3
        scaling = find_opt_scaling(gt_pts1, gt_pts2, pred_pts1, pred_pts2, valid1=valid1, valid2=valid2)
        
        # 3. render images(need gaussian model, camera, pipeline)
        rendered_images = []
        rendered_feats = []
        gt_images = []

        for i in range(len(pred)):
            # get gaussian model
            gaussians = GaussianModel.from_predictions(pred[i], sh_degree=3)
            # get camera
            ref_camera_extrinsics = gt1['camera_pose'][i]
            if target_views is None:
                target_view_list = [gt1, gt2, target_view]
            else:
                target_view_list = list(target_views)
            for j in range(len(target_view_list)):
                target_extrinsics = target_view_list[j]['camera_pose'][i]
                target_intrinsics = target_view_list[j]['camera_intrinsics'][i]
                image_shape = target_view_list[j]['true_shape'][i]
                scale = scaling[i]
                camera = get_scaled_camera(ref_camera_extrinsics, target_extrinsics, target_intrinsics, scale, image_shape)
                # render(image and features)
                rendered_output = render(camera, gaussians, self.pipeline, self.bg_color)
                rendered_images.append(rendered_output['render'])
                rendered_feats.append(rendered_output['feature_map'])
                gt_images.append(target_view_list[j]['img'][i] * 0.5 + 0.5)

        rendered_images = torch.stack(rendered_images, dim=0) # B, 3, H, W
        gt_images = torch.stack(gt_images, dim=0)
        rendered_feats = torch.stack(rendered_feats, dim=0) # B, d_feats, H, W
        rendered_feats = model.feature_expansion(rendered_feats) # B, 512, H//2, W//2
        gt_feats = model.lseg_feature_extractor.extract_features(gt_images) # B, 512, H//2, W//2
        image_loss = torch.abs(rendered_images - gt_images).mean()
        feature_loss = (1 - torch.nn.functional.cosine_similarity(rendered_feats, gt_feats, dim=1)).mean()
        loss = image_loss + self.feature_loss_weight * feature_loss

        return loss, {'image_loss': float(image_loss), 'feature_loss': float(feature_loss)}

class TestLoss(MultiLoss):
    def __init__(self, lables=LABELS):
        super().__init__()
        self.labels = lables
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.lpips_vgg = lpips.LPIPS(net='vgg')
        self.miou = JaccardIndex(num_classes=len(self.labels) + 1, task='multiclass', ignore_index=0)
        self.accuracy = Accuracy(num_classes=len(self.labels) + 1, task='multiclass', ignore_index=0)
        self.pipeline = DummyPipeline()
        self.register_buffer('bg_color', torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        self.register_buffer('color_map', COLORS.clone())

    def get_name(self):
        return 'TestLoss'

    def compute_loss(self, gt1, gt2, pred1, pred2, target_view=None, target_views=None, model=None, **kwargs):
        # 1. merge predictions
        pred = merge_and_split_predictions(pred1, pred2)

        # 2. calculate optimal scaling
        pred_pts1 = pred1['means']
        pred_pts2 = pred2['means']
        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'].to(in_camera1.device))  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt2['pts3d'].to(in_camera1.device))  # B,H,W,3
        scaling = find_opt_scaling(gt_pts1, gt_pts2, pred_pts1, pred_pts2, valid1=valid1, valid2=valid2)

        if target_views is None:
            if target_view is None:
                target_views = [gt1, gt2]
            else:
                # Keep backward compatibility while matching fd_3d's target_views protocol.
                target_views = [gt1, target_view, gt2]
        else:
            target_views = list(target_views)

        device = pred1['means'].device
        batch_size = len(pred)
        zero = torch.zeros((), device=device, dtype=torch.float32)

        # losses
        total_image_loss = zero.clone()
        total_feature_loss = zero.clone()
        # metrics
        total_psnr = zero.clone()
        total_ssim = zero.clone()
        total_lpips = zero.clone()
        total_miou = zero.clone()
        total_accuracy = zero.clone()
        total_rel = zero.clone()
        total_tau = zero.clone()
        total_lseg_iou = zero.clone()
        total_lseg_accuracy = zero.clone()
        total_pred_dataset_iou = zero.clone()
        total_pred_dataset_accuracy = zero.clone()

        results = None
        metric_views_per_sample = len(target_views) if len(target_views) <= 2 else (len(target_views) - 2)

        for i in range(batch_size):
            gaussians = GaussianModel.from_predictions(pred[i], sh_degree=3)
            ref_camera_extrinsics = gt1['camera_pose'][i]

            for target_idx, cur_target in enumerate(target_views):
                target_extrinsics = cur_target['camera_pose'][i]
                target_intrinsics = cur_target['camera_intrinsics'][i]
                image_shape = cur_target['true_shape'][i]

                camera = get_scaled_camera(
                    ref_camera_extrinsics,
                    target_extrinsics,
                    target_intrinsics,
                    scaling[i],
                    image_shape,
                )

                rendered_output = render(camera, gaussians, self.pipeline, self.bg_color)
                pr_imgs = torch.clamp(rendered_output['render'][None], 0, 1)
                pr_fmaps = model.feature_expansion(rendered_output['feature_map'][None])
                gt_imgs = cur_target['img'][i:i + 1] * 0.5 + 0.5
                gt_fmaps = model.lseg_feature_extractor.extract_features(gt_imgs)

                image_loss = torch.abs(pr_imgs - gt_imgs).mean()
                feature_loss = (1 - torch.nn.functional.cosine_similarity(pr_fmaps, gt_fmaps, dim=1)).mean()
                total_image_loss += image_loss
                total_feature_loss += feature_loss

                should_score = len(target_views) <= 2 or (target_idx != 0 and target_idx != len(target_views) - 1)
                if not should_score:
                    continue

                with torch.no_grad():
                    psnr_value = self.psnr(pr_imgs, gt_imgs)
                    ssim_value = self.ssim(pr_imgs, gt_imgs)
                    lpips_value = self.lpips_vgg(pr_imgs, gt_imgs).mean()

                    logits_per_image = model.lseg_feature_extractor.decode_feature(pr_fmaps, labelset=self.labels)
                    predict_label = torch.argmax(logits_per_image, dim=1) + 1

                    lseg_logits = model.lseg_feature_extractor.decode_feature(gt_fmaps, labelset=self.labels)
                    lseg_label = torch.argmax(lseg_logits, dim=1) + 1

                    dataset_label = _to_bhw(cur_target['labelmap'][i:i + 1]).long().to(device)
                    if dataset_label.shape[-2:] != predict_label.shape[-2:]:
                        dataset_label = F.interpolate(
                            dataset_label.unsqueeze(1).float(),
                            size=predict_label.shape[-2:],
                            mode='nearest',
                        ).squeeze(1).long()

                    iou_score = self.miou(predict_label, lseg_label)
                    accuracy = self.accuracy(predict_label, lseg_label)

                    pred_dataset_iou = self.miou(predict_label, dataset_label)
                    pred_dataset_accuracy = self.accuracy(predict_label, dataset_label)
                    lseg_iou = self.miou(lseg_label, dataset_label)
                    lseg_accuracy = self.accuracy(lseg_label, dataset_label)

                    pred_depth = _to_bchw(rendered_output['depth'][None]).float()
                    gt_depth = _to_bchw(cur_target['depthmap'][i:i + 1]).float().to(device)
                    if gt_depth.shape[-2:] != pred_depth.shape[-2:]:
                        gt_depth = F.interpolate(gt_depth, size=pred_depth.shape[-2:], mode='nearest')
                    rel, tau = calculate_depth_metrics(pred_depth, gt_depth)

                    total_psnr += psnr_value
                    total_ssim += ssim_value
                    total_lpips += lpips_value
                    total_miou += iou_score
                    total_accuracy += accuracy
                    total_rel += rel
                    total_tau += tau
                    total_pred_dataset_iou += pred_dataset_iou
                    total_pred_dataset_accuracy += pred_dataset_accuracy
                    total_lseg_iou += lseg_iou
                    total_lseg_accuracy += lseg_accuracy

                    middle_idx = len(target_views) // 2
                    if target_idx == middle_idx and i == batch_size // 2:
                        pr_mask = rearrange(self.color_map[predict_label], 'b h w c -> b c h w')
                        lseg_mask = rearrange(self.color_map[lseg_label], 'b h w c -> b c h w')
                        dataset_mask = rearrange(self.color_map[dataset_label], 'b h w c -> b c h w')
                        results = {
                            'rgb_gt': gt_imgs.detach(),
                            'rgb_ours': pr_imgs.detach(),
                            'sem_gt': dataset_mask.detach(),
                            'sem_lseg': lseg_mask.detach(),
                            'sem_ours': pr_mask.detach(),
                        }

        loss_normalizer = max(batch_size * len(target_views), 1)
        metric_normalizer = max(batch_size * metric_views_per_sample, 1)

        mean_image_loss = total_image_loss / loss_normalizer
        mean_feature_loss = total_feature_loss / loss_normalizer
        loss = mean_image_loss + mean_feature_loss

        mean_psnr = total_psnr / metric_normalizer
        mean_ssim = total_ssim / metric_normalizer
        mean_lpips = total_lpips / metric_normalizer
        mean_miou = total_miou / metric_normalizer
        mean_accuracy = total_accuracy / metric_normalizer
        mean_rel = total_rel / metric_normalizer
        mean_tau = total_tau / metric_normalizer
        mean_pred_dataset_iou = total_pred_dataset_iou / metric_normalizer
        mean_pred_dataset_accuracy = total_pred_dataset_accuracy / metric_normalizer
        mean_lseg_iou = total_lseg_iou / metric_normalizer
        mean_lseg_accuracy = total_lseg_accuracy / metric_normalizer

        return (
            loss,
            dict(
                image_loss=mean_image_loss.item(),
                feature_loss=mean_feature_loss.item(),
                mean_psnr=mean_psnr.item(),
                mean_ssim=mean_ssim.item(),
                mean_lpips=mean_lpips.item(),
                mean_miou=mean_miou.item(),
                mean_accuracy=mean_accuracy.item(),
                mean_rel=mean_rel.item(),
                mean_tau=mean_tau.item(),
                mean_pred_dataset_iou=mean_pred_dataset_iou.item(),
                mean_pred_dataset_accuracy=mean_pred_dataset_accuracy.item(),
                mean_lseg_iou=mean_lseg_iou.item(),
                mean_lseg_accuracy=mean_lseg_accuracy.item(),
                results=results,
            ),
        )

# loss for one batch
def loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, view2, target_view = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric((view1, view2))
        target_view, _ = make_batch_symmetric((target_view, target_view))
    # Get the actual model if it's distributed
    actual_model = model.module if hasattr(model, 'module') else model

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        pred1, pred2 = actual_model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(view1, view2, pred1, pred2, target_view=target_view, model=actual_model) if criterion is not None else None

    result = dict(view1=view1, view2=view2, target_view=target_view, pred1=pred1, pred2=pred2, loss=loss)
    return result[ret] if ret else result
