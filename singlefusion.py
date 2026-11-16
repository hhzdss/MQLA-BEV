from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from torch import nn
from torch.nn import functional as F

from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.registry import DATASETS, HOOKS, MODELS, TRANSFORMS
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from projects.BEVFusion.bevfusion import BEVFusion


@DATASETS.register_module()
class PohangDataset(Det3DDataset):
    METAINFO = dict(classes=('Vessel',))

    def parse_data_info(self, info: dict) -> dict:
        data_info = super().parse_data_info(info)
        data_info.setdefault('timestamp', 0.0)
        return data_info

    def parse_ann_info(self, info):
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            return None
        boxes = np.asarray(ann_info['gt_bboxes_3d'], dtype=np.float32)
        if boxes.shape[-1] == 7:
            boxes = np.concatenate(
                (boxes, np.zeros((boxes.shape[0], 2), dtype=np.float32)), axis=-1)
        ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(
            boxes, box_dim=9, origin=(0.5, 0.5, 0.5))
        return ann_info


@DATASETS.register_module()
class PohangDatasetKeepEmpty(PohangDataset):
    def parse_ann_info(self, info):
        ann_info = super().parse_ann_info(info)
        if ann_info is not None:
            return ann_info
        return dict(
            gt_bboxes_3d=LiDARInstance3DBoxes(
                np.zeros((0, 9), dtype=np.float32), box_dim=9,
                origin=(0.5, 0.5, 0.5)),
            gt_labels_3d=np.zeros((0,), dtype=np.int64))


def _group_norm(channels: int, max_groups: int = 16) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


@MODELS.register_module()
class PohangViewQualityEncoder(nn.Module):
    """Predict an independent quality scale for every camera view.

    The input consists of four cheap, content-derived statistics per view:
    mean absolute response, standard deviation, horizontal variation and
    vertical variation.  The last layer starts at zero, hence
    ``2 * sigmoid(0) == 1`` and every valid view initially follows baseline_2.
    """

    def __init__(self, hidden_channels: int = 16,
                 enabled: bool = True) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError('hidden_channels must be positive')
        self.enabled = bool(enabled)
        self.network = nn.Sequential(
            nn.Linear(4, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def image_statistics(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError('images must have shape [B, V, C, H, W]')
        mean_abs = images.abs().mean(dim=(2, 3, 4))
        std = images.float().std(dim=(2, 3, 4), unbiased=False).to(images.dtype)
        grad_x = (images[..., 1:] - images[..., :-1]).abs().mean(
            dim=(2, 3, 4))
        grad_y = (images[..., 1:, :] - images[..., :-1, :]).abs().mean(
            dim=(2, 3, 4))
        return torch.stack((mean_abs, std, grad_x, grad_y), dim=-1)

    def forward(self, images: torch.Tensor,
                valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        statistics = self.image_statistics(images)
        logits = self.network(statistics).squeeze(-1)
        if self.enabled:
            scales = 2. * logits.sigmoid()
        else:
            scales = torch.ones_like(logits)
        mask = valid_mask.to(device=images.device, dtype=images.dtype)
        scales = scales * mask
        return scales, logits


@MODELS.register_module()
class PohangQualityAwareAlignConvFuser(nn.Module):
    """Quality-weighted direct fusion with local cross-attention."""

    CAMERA_DEGRADATIONS = frozenset({
        'paired_rain', 'paired_fog', 'motion_blur', 'brightness',
        'darkness', 'camera_dropout', 'spatial_misalignment',
    })
    LIDAR_DEGRADATIONS = frozenset({
        'paired_rain', 'paired_fog', 'point_drop', 'beam_drop',
        'motion_blur',
    })

    def __init__(
            self,
            in_channels: Sequence[int] = (80, 64),
            out_channels: int = 64,
            quality_channels: int = 16,
            pool_factor: int = 4,
            lidar_prior: float = 0.05,
            use_camera_quality: bool = True,
            use_lidar_quality: bool = True,
            use_disagreement: bool = True,
            use_lidar_density: bool = True,
            use_ring_coverage: bool = True,
            use_camera_availability: bool = True,
            hard_camera_fallback: bool = True,
            fallback_mode: str = 'raw_lidar',
            use_masked_reconstruction: bool = True,
            reconstruction_mask_ratio: float = 0.15,
            use_modality_aux_heads: bool = True,
            use_local_cross_attention: bool = False,
            local_attention_radius: int = 2,
            quality_loss_weight: float = 0.10,
            reconstruction_loss_weight: float = 0.05,
            modality_aux_loss_weight: float = 0.10,
            freeze_loaded_bn_stats: bool = True) -> None:
        super().__init__()
        if len(in_channels) != 2 or any(channel <= 0 for channel in in_channels):
            raise ValueError('in_channels must contain [camera, lidar] sizes')
        if out_channels != in_channels[1]:
            raise ValueError('out_channels must equal LiDAR channels')
        if quality_channels <= 0 or pool_factor <= 0:
            raise ValueError('quality_channels and pool_factor must be positive')
        if local_attention_radius < 0:
            raise ValueError('local_attention_radius must be nonnegative')
        if not 0. <= reconstruction_mask_ratio < 1.:
            raise ValueError('reconstruction_mask_ratio must be in [0, 1)')
        if fallback_mode not in {'raw_lidar', 'convfuser_lidar'}:
            raise ValueError(
                "fallback_mode must be 'raw_lidar' or 'convfuser_lidar'")

        self.in_channels = tuple(int(value) for value in in_channels)
        self.out_channels = int(out_channels)
        self.quality_channels = int(quality_channels)
        self.pool_factor = int(pool_factor)
        self.lidar_prior = float(lidar_prior)
        self.use_camera_quality = bool(use_camera_quality)
        self.use_lidar_quality = bool(use_lidar_quality)
        self.use_disagreement = bool(use_disagreement)
        self.use_lidar_density = bool(use_lidar_density)
        self.use_ring_coverage = bool(use_ring_coverage)
        self.use_camera_availability = bool(use_camera_availability)
        self.hard_camera_fallback = bool(hard_camera_fallback)
        self.fallback_mode = fallback_mode
        self.use_masked_reconstruction = bool(use_masked_reconstruction)
        self.reconstruction_mask_ratio = float(reconstruction_mask_ratio)
        self.use_modality_aux_heads = bool(use_modality_aux_heads)
        self.use_local_cross_attention = bool(use_local_cross_attention)
        self.local_attention_radius = int(local_attention_radius)
        self.quality_loss_weight = float(quality_loss_weight)
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.modality_aux_loss_weight = float(modality_aux_loss_weight)
        self.freeze_loaded_bn_stats = bool(freeze_loaded_bn_stats)

        # Preserve baseline_2 ConvFuser parameter names exactly.
        self.add_module('0', nn.Conv2d(
            sum(self.in_channels), out_channels, 3, padding=1, bias=False))
        self.add_module('1', nn.BatchNorm2d(out_channels))
        self.add_module('2', nn.ReLU(inplace=True))

        camera_channels, lidar_channels = self.in_channels
        self.camera_projection = nn.Sequential(
            nn.Conv2d(camera_channels, quality_channels, 1, bias=False),
            _group_norm(quality_channels), nn.SiLU(inplace=True))
        self.lidar_projection = nn.Sequential(
            nn.Conv2d(lidar_channels, quality_channels, 1, bias=False),
            _group_norm(quality_channels), nn.SiLU(inplace=True))

        if self.use_local_cross_attention:
            # LiDAR queries a small neighborhood of aligned camera BEV tokens.
            # This attention runs at the pooled 0.8 m grid; its result is
            # projected to the ConvFuser width and added before SECOND.
            self.local_attention_query = nn.Conv2d(
                quality_channels, quality_channels, 1, bias=False)
            self.local_attention_key = nn.Conv2d(
                quality_channels, quality_channels, 1, bias=False)
            self.local_attention_value = nn.Conv2d(
                quality_channels, quality_channels, 1, bias=False)
            self.local_attention_output = nn.Conv2d(
                quality_channels, out_channels, 1, bias=False)
            self.local_attention_gate = nn.Parameter(torch.zeros(()))

        # Camera head receives projected camera/disagreement plus availability
        # and five global image/view-quality descriptors.
        self.camera_quality_head = nn.Sequential(
            nn.Conv2d(quality_channels * 2 + 6, quality_channels, 3,
                      padding=1, bias=False),
            _group_norm(quality_channels), nn.SiLU(inplace=True),
            nn.Conv2d(quality_channels, 1, 1))
        # LiDAR head receives projected LiDAR/disagreement, density and rings.
        self.lidar_quality_head = nn.Sequential(
            nn.Conv2d(quality_channels * 2 + 2, quality_channels, 3,
                      padding=1, bias=False),
            _group_norm(quality_channels), nn.SiLU(inplace=True),
            nn.Conv2d(quality_channels, 1, 1))
        nn.init.zeros_(self.camera_quality_head[-1].weight)
        nn.init.zeros_(self.camera_quality_head[-1].bias)
        nn.init.zeros_(self.lidar_quality_head[-1].weight)
        nn.init.zeros_(self.lidar_quality_head[-1].bias)

        self.reconstruction_decoder = nn.Sequential(
            nn.Conv2d(quality_channels * 2 + 1, quality_channels, 3,
                      padding=1, bias=False),
            _group_norm(quality_channels), nn.SiLU(inplace=True),
            nn.Conv2d(quality_channels, quality_channels, 3, padding=1))
        self.camera_aux_head = nn.Sequential(
            nn.Conv2d(quality_channels, quality_channels, 3,
                      padding=1, bias=False),
            _group_norm(quality_channels), nn.SiLU(inplace=True),
            nn.Conv2d(quality_channels, 1, 1))
        self.lidar_aux_head = nn.Sequential(
            nn.Conv2d(quality_channels, quality_channels, 3,
                      padding=1, bias=False),
            _group_norm(quality_channels), nn.SiLU(inplace=True),
            nn.Conv2d(quality_channels, 1, 1))
        nn.init.constant_(self.camera_aux_head[-1].bias, -2.19)
        nn.init.constant_(self.lidar_aux_head[-1].bias, -2.19)

        self._last_quality_logits: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._last_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._last_reconstruction_loss: Optional[torch.Tensor] = None
        self._last_aux_logits: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_loaded_bn_stats:
            self[1].eval()
        return self

    def __getitem__(self, index: int) -> nn.Module:
        module = self._modules.get(str(index))
        if module is None:
            raise IndexError(index)
        return module

    @staticmethod
    def _availability(camera_valid_mask: torch.Tensor,
                      reference: torch.Tensor) -> torch.Tensor:
        mask = torch.as_tensor(camera_valid_mask, device=reference.device)
        if mask.ndim != 2 or mask.shape[0] != reference.shape[0] or mask.shape[1] == 0:
            raise ValueError('camera_valid_mask must have shape [B, V]')
        return mask.to(reference.dtype).mean(dim=1).view(-1, 1, 1, 1)

    def _pool(self, value: torch.Tensor) -> torch.Tensor:
        if self.pool_factor == 1:
            return value
        return F.avg_pool2d(value, self.pool_factor, self.pool_factor)

    def _local_cross_attention(self, lidar_projected: torch.Tensor,
                               camera_projected: torch.Tensor,
                               camera_validity: torch.Tensor
                               ) -> torch.Tensor:
        """LiDAR-query local cross-attention over aligned camera BEV tokens."""
        batch, channels, height, width = lidar_projected.shape
        radius = self.local_attention_radius
        kernel_size = 2 * radius + 1

        query = self.local_attention_query(lidar_projected)
        key = self.local_attention_key(camera_projected)
        value = self.local_attention_value(camera_projected)
        key_windows = F.unfold(
            key, kernel_size=kernel_size, padding=radius).reshape(
                batch, channels, kernel_size * kernel_size, height, width)
        value_windows = F.unfold(
            value, kernel_size=kernel_size, padding=radius).reshape(
                batch, channels, kernel_size * kernel_size, height, width)

        # The BEV validity proxy is pooled from nonzero aligned camera cells;
        # combine it with the boundary mask so attention cannot select empty
        # BEV cells or positions outside the local window.
        valid_windows = F.unfold(
            camera_validity.to(camera_projected.dtype),
            kernel_size=kernel_size,
            padding=radius).reshape(
                batch, kernel_size * kernel_size, height, width) > 0
        in_bounds = F.unfold(
            camera_projected.new_ones((batch, 1, height, width)),
            kernel_size=kernel_size,
            padding=radius).reshape(
                batch, kernel_size * kernel_size, height, width) > 0
        valid_windows = valid_windows & in_bounds
        scores = (query.unsqueeze(2) * key_windows).sum(dim=1)
        scores = scores * (channels ** -0.5)
        scores = scores.masked_fill(~valid_windows, -1e4)
        attention = torch.softmax(scores, dim=1)
        attention = attention * valid_windows.to(attention.dtype)
        attention = attention / attention.sum(
            dim=1, keepdim=True).clamp_min(1e-6)
        return (attention.unsqueeze(1) * value_windows).sum(dim=2)

    @staticmethod
    def _validate_map(value: Optional[torch.Tensor], name: str,
                      reference: torch.Tensor) -> torch.Tensor:
        if value is None:
            return reference.new_zeros(
                (reference.shape[0], 1, *reference.shape[2:]))
        if (value.ndim != 4 or value.shape[0] != reference.shape[0]
                or value.shape[1] != 1
                or value.shape[2:] != reference.shape[2:]):
            raise ValueError(f'{name} must have shape [B, 1, H, W]')
        return value.to(device=reference.device, dtype=reference.dtype)

    def _masked_reconstruction(self, camera_projected: torch.Tensor,
                               lidar_projected: torch.Tensor) -> None:
        self._last_reconstruction_loss = camera_projected.new_zeros(())
        if (not self.training or not self.use_masked_reconstruction
                or self.reconstruction_mask_ratio <= 0):
            return
        keep = (torch.rand(
            camera_projected.shape[0], 1, *camera_projected.shape[2:],
            device=camera_projected.device) >= self.reconstruction_mask_ratio)
        keep = keep.to(camera_projected.dtype)
        decoder_input = torch.cat(
            (camera_projected * keep, lidar_projected, keep), dim=1)
        reconstructed = self.reconstruction_decoder(decoder_input)
        missing = 1. - keep
        denominator = missing.sum().clamp_min(1.) * camera_projected.shape[1]
        self._last_reconstruction_loss = (
            (reconstructed - camera_projected.detach()).abs()
            * missing).sum() / denominator

    def forward(self,
                inputs: Sequence[torch.Tensor],
                camera_valid_mask: Optional[torch.Tensor] = None,
                lidar_density: Optional[torch.Tensor] = None,
                lidar_ring_coverage: Optional[torch.Tensor] = None,
                camera_quality_features: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError('Expected exactly [camera_features, lidar_features]')
        camera, lidar = inputs
        for feature, channels, name in zip(
                inputs, self.in_channels, ('camera', 'lidar')):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError(f'{name} must have {channels} BCHW channels')
        if camera.shape[0] != lidar.shape[0] or camera.shape[2:] != lidar.shape[2:]:
            raise ValueError('Camera and LiDAR batch/spatial shapes must match')

        if camera_valid_mask is None:
            camera_valid_mask = camera.new_ones((camera.shape[0], 1))
        availability = self._availability(camera_valid_mask, camera)
        density = self._validate_map(lidar_density, 'lidar_density', lidar)
        if not self.use_lidar_density:
            density = torch.zeros_like(density)
        density_low = self._pool(density)

        if lidar_ring_coverage is None:
            ring_coverage = camera.new_ones((camera.shape[0], 1, 1, 1))
        else:
            ring_coverage = lidar_ring_coverage.to(
                device=camera.device, dtype=camera.dtype).reshape(-1, 1, 1, 1)
        if not self.use_ring_coverage:
            ring_coverage = torch.ones_like(ring_coverage)

        if camera_quality_features is None:
            camera_quality_features = camera.new_zeros((camera.shape[0], 5))
        if camera_quality_features.shape != (camera.shape[0], 5):
            raise ValueError('camera_quality_features must have shape [B, 5]')
        camera_quality_features = camera_quality_features.to(
            device=camera.device, dtype=camera.dtype)

        lidar_low = self._pool(lidar)
        camera_projected = self.camera_projection(self._pool(camera))
        lidar_projected = self.lidar_projection(lidar_low)
        if self.use_local_cross_attention:
            camera_validity = self._pool(
                (camera.abs().sum(dim=1, keepdim=True) > 0).to(
                    camera.dtype)) > 0
        disagreement = (camera_projected - lidar_projected).abs()
        if not self.use_disagreement:
            disagreement = torch.zeros_like(disagreement)
        low_shape = camera_projected.shape[2:]
        availability_map = availability.expand(-1, -1, *low_shape)
        ring_map = ring_coverage.expand(-1, -1, *low_shape)
        image_map = camera_quality_features[:, :, None, None].expand(
            -1, -1, *low_shape)

        camera_quality_input = torch.cat(
            (camera_projected, disagreement,
             availability_map, image_map), dim=1)
        lidar_quality_input = torch.cat(
            (lidar_projected, disagreement, density_low, ring_map), dim=1)
        camera_logits = self.camera_quality_head(camera_quality_input)
        lidar_logits = self.lidar_quality_head(lidar_quality_input)
        if not self.use_camera_quality:
            camera_logits = torch.zeros_like(camera_logits)
        if not self.use_lidar_quality:
            lidar_logits = torch.zeros_like(lidar_logits)

        if self.use_camera_availability:
            camera_logits = camera_logits + availability.clamp_min(1e-4).log()
        lidar_logits = lidar_logits + self.lidar_prior
        weights = torch.softmax(torch.cat((camera_logits, lidar_logits), dim=1),
                                dim=1)
        camera_weight, lidar_weight = weights[:, :1], weights[:, 1:]
        self._last_quality_logits = (camera_logits, lidar_logits)
        self._last_weights = (camera_weight, lidar_weight)

        camera_scale = F.interpolate(
            2. * camera_weight, size=camera.shape[2:], mode='bilinear',
            align_corners=False)
        lidar_scale = F.interpolate(
            2. * lidar_weight, size=lidar.shape[2:], mode='bilinear',
            align_corners=False)
        candidate = self[2](self[1](self[0](torch.cat(
            (camera * camera_scale, lidar * lidar_scale), dim=1))))

        if self.hard_camera_fallback:
            has_camera = (availability > 0).expand_as(candidate)
            if self.fallback_mode == 'raw_lidar':
                fallback = lidar
            else:
                # The downstream SECOND/FPN/CenterHead was fine-tuned on the
                # original ConvFuser output, not on raw PointPillars BEV.
                # Preserve that feature interface while giving the fuser only
                # LiDAR information when every camera is unavailable.
                fallback = self[2](self[1](self[0](torch.cat(
                    (torch.zeros_like(camera), lidar), dim=1))))
            candidate = torch.where(has_camera, candidate, fallback)

        if self.use_local_cross_attention:
            attention_context = self._local_cross_attention(
                lidar_projected, camera_projected, camera_validity)
            # The learned quality map limits camera context before the
            # zero-initialized residual gate adds it to the ConvFuser output.
            attention_context = attention_context * camera_weight
            attention_residual = self.local_attention_output(attention_context)
            attention_residual = F.interpolate(
                attention_residual, size=candidate.shape[2:],
                mode='bilinear', align_corners=False)
            has_camera = (availability > 0).to(candidate.dtype)
            candidate = candidate + (
                self.local_attention_gate.tanh().to(candidate.dtype)
                * attention_residual * has_camera)

        self._masked_reconstruction(camera_projected, lidar_projected)
        if self.use_modality_aux_heads:
            self._last_aux_logits = (
                self.camera_aux_head(camera_projected),
                self.lidar_aux_head(lidar_projected))
        else:
            self._last_aux_logits = None
        return candidate

    @classmethod
    def quality_targets(cls, metas: Sequence[Mapping],
                        reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        camera_targets, lidar_targets = [], []
        severity_quality = {1: .8, 2: .6, 3: .4}
        for meta in metas:
            camera_quality = 1.
            lidar_quality = 1.
            corruption = meta.get('maritime_corruption', {}) or {}
            events = corruption.get('events', [])
            if corruption.get('type') == 'camera_dropout' and not events:
                events = [corruption]
            for event in events:
                name = event.get('type')
                severity = event.get('severity', 0)
                if severity == 'all':
                    event_quality = 0.
                else:
                    event_quality = severity_quality.get(int(severity or 0), 1.)
                if name == 'camera_dropout' and 'camera_valid_mask' in event:
                    event_quality = float(np.asarray(
                        event['camera_valid_mask'], dtype=np.float32).mean())
                if name in cls.CAMERA_DEGRADATIONS:
                    camera_quality = min(camera_quality, event_quality)
                if name in cls.LIDAR_DEGRADATIONS:
                    lidar_quality = min(lidar_quality, event_quality)
            camera_targets.append(camera_quality)
            lidar_targets.append(lidar_quality)
        return (reference.new_tensor(camera_targets),
                reference.new_tensor(lidar_targets))

    @staticmethod
    def _centre_heatmap(batch_data_samples: List[Det3DDataSample],
                        shape: Tuple[int, int], reference: torch.Tensor,
                        point_cloud_range: Sequence[float]) -> torch.Tensor:
        height, width = shape
        target = reference.new_zeros((len(batch_data_samples), 1, height, width))
        x_min, y_min, _, x_max, y_max, _ = point_cloud_range
        for batch_index, sample in enumerate(batch_data_samples):
            instances = getattr(sample, 'gt_instances_3d', None)
            boxes = getattr(instances, 'bboxes_3d', None) if instances is not None else None
            if boxes is None:
                continue
            if len(boxes) == 0:
                continue
            centers = boxes.tensor[:, :2].to(reference.device)
            cols = torch.floor((centers[:, 0] - x_min) /
                               (x_max - x_min) * width).long()
            rows = torch.floor((centers[:, 1] - y_min) /
                               (y_max - y_min) * height).long()
            valid = ((rows >= 0) & (rows < height)
                     & (cols >= 0) & (cols < width))
            target[batch_index, 0, rows[valid], cols[valid]] = 1.
        return target

    @staticmethod
    def _centre_focal_loss(logits: torch.Tensor,
                           target: torch.Tensor) -> torch.Tensor:
        probability = logits.sigmoid().clamp(1e-4, 1. - 1e-4)
        positive = -torch.log(probability) * (1. - probability).pow(2) * target
        negative = -torch.log(1. - probability) * probability.pow(2) * (1. - target)
        normalizer = target.sum().clamp_min(1.)
        return (positive.sum() + negative.sum()) / normalizer

    def auxiliary_losses(self,
                         batch_data_samples: List[Det3DDataSample],
                         metas: Sequence[Mapping],
                         point_cloud_range: Sequence[float],
                         view_quality_logits: Optional[torch.Tensor] = None
                         ) -> Dict[str, torch.Tensor]:
        if self._last_quality_logits is None or self._last_weights is None:
            raise RuntimeError('forward must run before auxiliary_losses')
        camera_logits, lidar_logits = self._last_quality_logits
        camera_global = camera_logits.mean(dim=(1, 2, 3))
        lidar_global = lidar_logits.mean(dim=(1, 2, 3)) - self.lidar_prior
        camera_target, lidar_target = self.quality_targets(metas, camera_global)
        quality_loss = (
            F.binary_cross_entropy_with_logits(camera_global, camera_target)
            + F.binary_cross_entropy_with_logits(lidar_global, lidar_target))
        if view_quality_logits is not None:
            view_target = camera_target[:, None].expand_as(view_quality_logits)
            quality_loss = quality_loss + F.binary_cross_entropy_with_logits(
                view_quality_logits, view_target)

        losses: Dict[str, torch.Tensor] = {
            'loss_sensor_quality': self.quality_loss_weight * quality_loss,
        }
        if self._last_reconstruction_loss is not None:
            losses['loss_masked_bev_reconstruction'] = (
                self.reconstruction_loss_weight
                * self._last_reconstruction_loss)
        if self._last_aux_logits is not None:
            camera_aux, lidar_aux = self._last_aux_logits
            target = self._centre_heatmap(
                batch_data_samples, tuple(camera_aux.shape[2:]), camera_aux,
                point_cloud_range)
            losses['loss_camera_aux_center'] = (
                self.modality_aux_loss_weight
                * self._centre_focal_loss(camera_aux, target))
            losses['loss_lidar_aux_center'] = (
                self.modality_aux_loss_weight
                * self._centre_focal_loss(lidar_aux, target))

        camera_weight, lidar_weight = self._last_weights
        losses['quality_camera_weight'] = camera_weight.detach().mean()
        losses['quality_lidar_weight'] = lidar_weight.detach().mean()
        return losses


@MODELS.register_module()
class PohangQualityAwareDirectBEVFusion(BEVFusion):
    """Quality-aware fusion over the calibrated camera BEV."""

    def __init__(self,
                 point_cloud_range: Sequence[float],
                 expected_lidar_beams: int = 64,
                 density_cap: int = 20,
                 view_availability_loss_weight: float = 0.05,
                 view_quality_encoder: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if len(point_cloud_range) != 6:
            raise ValueError('point_cloud_range must contain six values')
        if expected_lidar_beams <= 0 or density_cap <= 0:
            raise ValueError('expected_lidar_beams and density_cap must be positive')
        self.robust_point_cloud_range = tuple(float(v) for v in point_cloud_range)
        self.expected_lidar_beams = int(expected_lidar_beams)
        self.density_cap = int(density_cap)
        if view_availability_loss_weight < 0:
            raise ValueError('view_availability_loss_weight must be nonnegative')
        self.view_availability_loss_weight = float(view_availability_loss_weight)
        view_quality_encoder = view_quality_encoder or dict(
            type='PohangViewQualityEncoder')
        self.view_quality_encoder = MODELS.build(view_quality_encoder)
        self._last_view_quality_logits: Optional[torch.Tensor] = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        translated = state_dict.copy()
        for key, value in state_dict.items():
            if key.startswith('pts_bbox_head.'):
                translated.setdefault('bbox_head.' + key[len('pts_bbox_head.'):], value)
                translated.pop(key)
        return super().load_state_dict(translated, strict=strict, assign=assign)

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            voxel_list, coord_list, num_points_list = [], [], []
            for batch_index, point_tensor in enumerate(points):
                voxels, coordinates, num_points = self.pts_voxel_layer(point_tensor)
                voxel_list.append(voxels)
                coord_list.append(torch.nn.functional.pad(
                    coordinates, (1, 0), mode='constant', value=batch_index))
                num_points_list.append(num_points)
            if not voxel_list:
                raise RuntimeError('Pillar BEVFusion received an empty point batch')
            voxels = torch.cat(voxel_list, dim=0)
            xyz_coordinates = torch.cat(coord_list, dim=0)
            num_points = torch.cat(num_points_list, dim=0)
            pillar_coordinates = xyz_coordinates[:, [0, 3, 2, 1]].contiguous()
        pillar_features = self.pts_voxel_encoder(
            voxels, num_points, pillar_coordinates)
        return self.pts_middle_encoder(
            pillar_features, pillar_coordinates, batch_size=len(points))

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        features = self.extract_feat(
            batch_inputs_dict,
            [sample.metainfo for sample in batch_data_samples])
        outputs = self.bbox_head.predict(features, batch_data_samples)
        return self.add_pred_to_datasample(batch_data_samples, outputs)

    @staticmethod
    def _camera_mask(batch_input_metas: Sequence[Mapping], views: int,
                     reference: torch.Tensor) -> torch.Tensor:
        masks = []
        for meta in batch_input_metas:
            value = meta.get('camera_valid_mask', np.ones(views, dtype=np.bool_))
            mask = torch.as_tensor(value, device=reference.device, dtype=torch.bool)
            if mask.ndim != 1 or mask.numel() != views:
                raise ValueError(f'camera_valid_mask must contain {views} flags')
            masks.append(mask)
        return torch.stack(masks, dim=0)

    def _lidar_quality_inputs(self, points: Sequence[torch.Tensor],
                              shape: Tuple[int, int],
                              reference: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = shape
        x_min, y_min, _, x_max, y_max, _ = self.robust_point_cloud_range
        x_step = (x_max - x_min) / width
        y_step = (y_max - y_min) / height
        density = reference.new_zeros((len(points), 1, height, width))
        ring_coverage = reference.new_zeros((len(points), 1))
        for batch_index, point_value in enumerate(points):
            tensor = point_value.tensor if hasattr(point_value, 'tensor') else point_value
            tensor = tensor.to(reference.device)
            valid_non_padding = torch.any(tensor[:, :3] != 0, dim=1)
            xyz = tensor[:, :3]
            x_index = torch.floor((xyz[:, 0] - x_min) / x_step).long()
            y_index = torch.floor((xyz[:, 1] - y_min) / y_step).long()
            valid = (valid_non_padding & (x_index >= 0) & (x_index < width)
                     & (y_index >= 0) & (y_index < height))
            flat = y_index[valid] * width + x_index[valid]
            counts = torch.bincount(flat, minlength=height * width)
            density[batch_index, 0] = counts.reshape(height, width).to(
                reference.dtype)
            if tensor.shape[1] >= 5 and torch.any(valid_non_padding):
                rings = torch.unique(tensor[valid_non_padding, 4])
                ring_coverage[batch_index, 0] = min(
                    float(rings.numel()) / self.expected_lidar_beams, 1.)
            else:
                ring_coverage[batch_index, 0] = 1.
        density.clamp_(max=self.density_cap)
        density = torch.log1p(density) / np.log1p(self.density_cap)
        return density, ring_coverage

    @staticmethod
    def _camera_quality_features(images: torch.Tensor,
                                 view_scales: torch.Tensor,
                                 valid_mask: torch.Tensor) -> torch.Tensor:
        statistics = PohangViewQualityEncoder.image_statistics(images)
        valid = valid_mask.to(images.dtype)
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.)
        aggregated = (statistics * valid[..., None]).sum(dim=1) / denominator
        mean_scale = (view_scales * valid).sum(dim=1, keepdim=True) / denominator
        return torch.cat((aggregated, mean_scale), dim=1)

    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        imgs = batch_inputs_dict.get('imgs')
        points = batch_inputs_dict.get('points')
        if imgs is None or points is None:
            raise ValueError('Quality-aware BEVFusion requires images and LiDAR')
        imgs = imgs.contiguous()
        camera_mask = self._camera_mask(
            batch_input_metas, imgs.shape[1], imgs)
        view_scales, view_logits = self.view_quality_encoder(imgs, camera_mask)
        self._last_view_quality_logits = view_logits
        scaled_imgs = imgs * view_scales[:, :, None, None, None].to(imgs.dtype)
        camera_quality_features = self._camera_quality_features(
            imgs, view_scales, camera_mask)

        lidar2image, camera_intrinsics, camera2lidar = [], [], []
        img_aug_matrix, lidar_aug_matrix = [], []
        for meta in batch_input_metas:
            lidar2image.append(meta['lidar2img'])
            camera_intrinsics.append(meta['cam2img'])
            camera2lidar.append(meta['cam2lidar'])
            img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
            lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))

        camera_bev_xy = self.extract_img_feat(
            scaled_imgs, deepcopy(points),
            imgs.new_tensor(np.asarray(lidar2image)),
            imgs.new_tensor(np.asarray(camera_intrinsics)),
            imgs.new_tensor(np.asarray(camera2lidar)),
            imgs.new_tensor(np.asarray(img_aug_matrix)),
            imgs.new_tensor(np.asarray(lidar_aug_matrix)), batch_input_metas)
        camera_bev_yx = camera_bev_xy.transpose(2, 3).contiguous()
        lidar_bev_yx = self.extract_pts_feat(batch_inputs_dict)
        if camera_bev_yx.shape != (
                camera_bev_yx.shape[0], self.fusion_layer.in_channels[0],
                *lidar_bev_yx.shape[2:]):
            raise ValueError(
                'Camera and LiDAR BEV grids/channels do not match fuser config')

        density, ring_coverage = self._lidar_quality_inputs(
            points, tuple(lidar_bev_yx.shape[2:]), lidar_bev_yx)
        fused_bev = self.fusion_layer(
            [camera_bev_yx, lidar_bev_yx],
            camera_valid_mask=camera_mask,
            lidar_density=density,
            lidar_ring_coverage=ring_coverage,
            camera_quality_features=camera_quality_features)
        features = self.pts_neck(self.pts_backbone(fused_bev))
        if not isinstance(features, (list, tuple)) or len(features) != 1:
            raise ValueError('Quality-aware CenterPoint requires one FPN level')
        return features

    def loss(self, batch_inputs_dict: Dict[str, Optional[torch.Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs):
        metas = [sample.metainfo for sample in batch_data_samples]
        features = self.extract_feat(batch_inputs_dict, metas)
        losses = self.bbox_head.loss(features, batch_data_samples)
        losses.update(self.fusion_layer.auxiliary_losses(
            batch_data_samples, metas, self.robust_point_cloud_range,
            self._last_view_quality_logits))
        imgs = batch_inputs_dict.get('imgs')
        if imgs is None or self._last_view_quality_logits is None:
            raise RuntimeError('View quality logits were not produced')
        camera_mask = self._camera_mask(metas, imgs.shape[1], imgs)
        targets = camera_mask.to(self._last_view_quality_logits.dtype)
        losses['loss_view_availability'] = (
            self.view_availability_loss_weight
            * F.binary_cross_entropy_with_logits(
                self._last_view_quality_logits, targets))
        return losses


@MODELS.register_module()
class PohangQualityAwareSingleFusionNoSafety(
        PohangQualityAwareDirectBEVFusion):
    def __init__(self,
                 lidar_safety_expert=None,
                 lidar_safety_checkpoint=None,
                 freeze_lidar_safety: bool = False,
                 train_lidar_safety: bool = False,
                 **kwargs) -> None:
        if lidar_safety_expert is not None or lidar_safety_checkpoint is not None:
            raise ValueError('Safety expert settings must be None')
        if freeze_lidar_safety or train_lidar_safety:
            raise ValueError('Safety switches must be disabled')
        super().__init__(**kwargs)


@TRANSFORMS.register_module()
class PohangFusionRandomFlip3D(BaseTransform):
    """Reflect the BEV y axis while preserving camera/point alignment.

    With a front-facing camera and forward-only detection range, flipping the
    BEV x axis would move foreground objects behind the ego origin. It is
    therefore explicitly disabled. Image pixels remain unchanged: the view
    transform uses ``lidar_aug_matrix`` to map them into augmented LiDAR BEV.
    """

    def __init__(self,
                 flip_ratio_bev_horizontal: float = 0.5,
                 flip_ratio_bev_vertical: float = 0.0) -> None:
        if not 0 <= flip_ratio_bev_horizontal <= 1:
            raise ValueError('Horizontal flip probability must be in [0, 1]')
        if flip_ratio_bev_vertical != 0:
            raise ValueError('Vertical BEV flipping is disabled for Pohang')
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip = np.random.rand() < self.flip_ratio_bev_horizontal
        data['pcd_horizontal_flip'] = bool(flip)
        data['pcd_vertical_flip'] = False
        matrix = np.asarray(
            data.get('lidar_aug_matrix', np.eye(4, dtype=np.float32)))
        if matrix.shape != (4, 4):
            raise ValueError('lidar_aug_matrix must have shape (4, 4)')

        if flip:
            if 'points' in data:
                data['points'].flip('horizontal')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('horizontal')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, :, ::-1].copy()
            reflection = np.eye(4, dtype=matrix.dtype)
            reflection[1, 1] = -1
            # Left composition also reflects any previous translation.
            matrix = reflection @ matrix
            data.setdefault('transformation_3d_flow', []).append('HF')

        data['lidar_aug_matrix'] = matrix
        return data




@HOOKS.register_module()
class PohangFusionWarmupHook(Hook):
    """Temporarily freeze pretrained LiDAR modules after optimizer creation.

    Epochs are zero-based: ``warmup_epochs=2`` freezes epochs 0 and 1, then
    restores each parameter's original trainability from epoch 2 onward.
    The optimizer must be built with these parameters present. Do not mark
    these modules frozen in the model config or optimizer constructor.

    DDP requires ``find_unused_parameters=True`` and ``static_graph=False``
    because the set of trainable parameters changes after the warmup. Resume
    decisions use ``runner.epoch`` and need no separate persisted hook state.
    The new heading classifier is frozen with the head; its auxiliary loss is
    disabled during warmup so features do not fit a fixed random classifier.
    """

    def __init__(self,
                 warmup_epochs: int = 2,
                 module_names: Sequence[str] = (
                     'pts_middle_encoder', 'pts_backbone', 'pts_neck',
                     'bbox_head')) -> None:
        if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
            raise ValueError('warmup_epochs must be a nonnegative integer')
        if not module_names or len(set(module_names)) != len(module_names):
            raise ValueError('module_names must be nonempty and unique')
        self.warmup_epochs = warmup_epochs
        self.module_names = tuple(module_names)
        self._modules = None
        self._original_requires_grad = None
        self._is_frozen = None
        self._last_logged_epoch = None
        self._heading_head = None
        self._original_heading_weight = None

    def _initialize(self, runner) -> None:
        if self._modules is not None:
            return
        model = runner.model
        while is_model_wrapper(model):
            if isinstance(model, nn.parallel.DistributedDataParallel):
                if self.warmup_epochs and (
                        not model.find_unused_parameters
                        or getattr(model, 'static_graph', False)):
                    raise RuntimeError(
                        'PohangFusionWarmupHook needs DDP '
                        'find_unused_parameters=True and static_graph=False')
            model = model.module
        modules = []
        for name in self.module_names:
            module = getattr(model, name, None)
            if not isinstance(module, nn.Module):
                raise ValueError(f'Cannot freeze missing model module: {name}')
            modules.append(module)
        self._modules = modules
        self._original_requires_grad = {
            parameter: parameter.requires_grad
            for module in modules for parameter in module.parameters()
        }
        head = getattr(model, 'bbox_head', None)
        if 'bbox_head' in self.module_names and hasattr(
                head, 'heading_loss_weight'):
            self._heading_head = head
            self._original_heading_weight = head.heading_loss_weight

    def before_train_epoch(self, runner) -> None:
        self._initialize(runner)
        freeze = runner.epoch < self.warmup_epochs
        if self._heading_head is not None:
            self._heading_head.heading_loss_weight = (
                0.0 if freeze else self._original_heading_weight)
        if freeze != self._is_frozen:
            for parameter, was_trainable in (
                    self._original_requires_grad.items()):
                parameter.requires_grad_(False if freeze else was_trainable)
                # Avoid a stale gradient surviving a change of trainability.
                parameter.grad = None
            if not freeze:
                for module in self._modules:
                    module.train()
            self._is_frozen = freeze
        if freeze:
            self._set_frozen_eval()
        if self._last_logged_epoch != runner.epoch:
            state = 'frozen (eval)' if freeze else 'trainability restored'
            runner.logger.info(
                f'Pohang fusion epoch {runner.epoch + 1}: '
                f'{", ".join(self.module_names)} {state}')
            self._last_logged_epoch = runner.epoch

    def _set_frozen_eval(self) -> None:
        for module in self._modules:
            module.eval()

    def before_train_iter(self, runner, batch_idx: int,
                          data_batch=None) -> None:
        # EpochBasedTrainLoop can call model.train() after before_train_epoch;
        # applying eval here ensures frozen BN statistics do not keep drifting.
        if self._is_frozen:
            self._set_frozen_eval()




@HOOKS.register_module()
class PohangFreezeBNStatsHook(Hook):
    """Keep pretrained image/fusion BN statistics fixed during corrupt tuning.

    Affine BN parameters remain trainable.  Only running mean/variance updates
    are disabled, preventing the explicit all-camera-dropout batches from
    replacing statistics learned by the clean SOTA checkpoint.
    """

    def __init__(self, module_names: Sequence[str] = (
            'img_neck', 'view_transform')) -> None:
        if not module_names or len(set(module_names)) != len(module_names):
            raise ValueError('module_names must be nonempty and unique')
        self.module_names = tuple(module_names)
        self._batch_norms = None

    def _initialize(self, runner) -> None:
        if self._batch_norms is not None:
            return
        model = runner.model
        while is_model_wrapper(model):
            model = model.module
        batch_norms = []
        for name in self.module_names:
            module = getattr(model, name, None)
            if not isinstance(module, nn.Module):
                raise ValueError(
                    f'Cannot freeze BN statistics for missing module: {name}')
            batch_norms.extend(child for child in module.modules()
                               if isinstance(child, nn.modules.batchnorm._BatchNorm))
        self._batch_norms = batch_norms

    def _set_eval(self, runner) -> None:
        self._initialize(runner)
        for batch_norm in self._batch_norms:
            batch_norm.eval()

    def before_train_epoch(self, runner) -> None:
        self._set_eval(runner)

    def before_train_iter(self, runner, batch_idx: int,
                          data_batch=None) -> None:
        # The training loop may call model.train() after the epoch callback.
        self._set_eval(runner)

