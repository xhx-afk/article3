"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register


@register()
class DEIMCriterion(nn.Module):
    """ This class computes the loss for DEIM.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        qcmr_encoder_quality_enabled=False,
        qcmr_encoder_quality_loss_weight=0.5,
        qcmr_encoder_quality_neg_weight=0.25,
        qcmr_encoder_quality_target_gamma=1.0,
        qcmr_local_rank_enabled=False,
        qcmr_rank_loss_weight=0.05,
        qcmr_rank_start_epoch=5,
        qcmr_rank_min_pos_iou=0.50,
        qcmr_rank_min_comp_iou=0.50,
        qcmr_rank_topk_competitors=3,
        qcmr_rank_better_comp_tolerance=0.05,
        qcmr_rank_temperature=0.20,
        qcmr_rank_pos_quality_power=1.0,
        qcmr_rank_comp_quality_power=1.0,
        qcmr_rank_eps=1e-6,
        ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.qcmr_encoder_quality_enabled = qcmr_encoder_quality_enabled
        self.qcmr_encoder_quality_loss_weight = qcmr_encoder_quality_loss_weight
        self.qcmr_encoder_quality_neg_weight = qcmr_encoder_quality_neg_weight
        self.qcmr_encoder_quality_target_gamma = qcmr_encoder_quality_target_gamma
        self.qcmr_local_rank_enabled = qcmr_local_rank_enabled
        self.qcmr_rank_loss_weight = qcmr_rank_loss_weight
        self.qcmr_rank_start_epoch = int(qcmr_rank_start_epoch)
        self.qcmr_rank_min_pos_iou = qcmr_rank_min_pos_iou
        self.qcmr_rank_min_comp_iou = qcmr_rank_min_comp_iou
        self.qcmr_rank_topk_competitors = int(qcmr_rank_topk_competitors)
        self.qcmr_rank_better_comp_tolerance = qcmr_rank_better_comp_tolerance
        self.qcmr_rank_temperature = qcmr_rank_temperature
        self.qcmr_rank_pos_quality_power = qcmr_rank_pos_quality_power
        self.qcmr_rank_comp_quality_power = qcmr_rank_comp_quality_power
        self.qcmr_rank_eps = qcmr_rank_eps
        self.qcmr_debug_stats = {}
        if self.qcmr_encoder_quality_enabled:
            if self.qcmr_encoder_quality_loss_weight < 0 or self.qcmr_encoder_quality_neg_weight < 0:
                raise ValueError('Encoder quality loss weights must be non-negative.')
            if self.qcmr_encoder_quality_target_gamma <= 0:
                raise ValueError('qcmr_encoder_quality_target_gamma must be positive.')
        if self.qcmr_local_rank_enabled:
            if self.qcmr_rank_loss_weight < 0 or self.qcmr_rank_start_epoch < 0:
                raise ValueError('QCCR rank loss weight and start epoch must be non-negative.')
            if not 0 <= self.qcmr_rank_min_pos_iou <= 1 or not 0 <= self.qcmr_rank_min_comp_iou <= 1:
                raise ValueError('QCCR IoU thresholds must be in [0, 1].')
            if self.qcmr_rank_topk_competitors <= 0 or self.qcmr_rank_temperature <= 0:
                raise ValueError('QCCR top-k and temperature must be positive.')
            if self.qcmr_rank_better_comp_tolerance < 0 or self.qcmr_rank_eps <= 0:
                raise ValueError('QCCR tolerance and eps must be non-negative/positive.')
            if self.qcmr_rank_pos_quality_power < 0 or self.qcmr_rank_comp_quality_power < 0:
                raise ValueError('QCCR quality powers must be non-negative.')

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    def _qcmr_quality_loss(self, outputs, targets, indices, stats_prefix=''):
        """Supervise a class-agnostic quality logit with matched-box IoU."""
        quality_logits = outputs['pred_quality']
        if quality_logits.shape[-1] != 1:
            raise ValueError('pred_quality must have shape [batch, queries, 1].')
        quality_logits = quality_logits.squeeze(-1)

        quality_targets = torch.zeros_like(quality_logits)
        positive_mask = torch.zeros_like(quality_logits, dtype=torch.bool)
        matched_iou_for_stats = quality_logits.new_empty(0)
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() > 0:
            src_boxes = outputs['pred_boxes'][idx].detach()
            target_boxes = torch.cat(
                [target['boxes'][target_idx] for target, (_, target_idx) in zip(targets, indices)],
                dim=0,
            )
            matched_iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            matched_iou = torch.diag(matched_iou).clamp(0, 1)
            matched_iou_for_stats = matched_iou
            quality_targets[idx] = matched_iou.pow(
                self.qcmr_encoder_quality_target_gamma).to(quality_targets.dtype)
            positive_mask[idx] = True

        elementwise_loss = F.binary_cross_entropy_with_logits(
            quality_logits, quality_targets, reduction='none')
        zero = quality_logits.sum() * 0.0
        positive_loss = elementwise_loss[positive_mask].mean() if positive_mask.any() else zero
        negative_mask = ~positive_mask
        negative_loss = elementwise_loss[negative_mask].mean() if negative_mask.any() else zero

        with torch.no_grad():
            quality_probs = quality_logits.float().sigmoid()
            stats_zero = quality_probs.sum() * 0.0
            positive_quality = quality_probs[idx]
            negative_quality = quality_probs[negative_mask]
            positive_mean = positive_quality.mean() if positive_quality.numel() else stats_zero
            negative_mean = negative_quality.mean() if negative_quality.numel() else stats_zero

            correlation = stats_zero
            if positive_quality.numel() >= 2:
                matched_iou_float = matched_iou_for_stats.float()
                quality_centered = positive_quality - positive_quality.mean()
                iou_centered = matched_iou_float - matched_iou_float.mean()
                denominator = torch.sqrt(
                    quality_centered.square().sum() * iou_centered.square().sum())
                eps = torch.finfo(quality_centered.dtype).eps
                correlation = torch.where(
                    denominator > eps,
                    (quality_centered * iou_centered).sum() / denominator.clamp_min(eps),
                    stats_zero,
                )

            stats = {
                f'qcmr_{stats_prefix}quality_pos_mean': positive_mean.detach(),
                f'qcmr_{stats_prefix}quality_neg_mean': negative_mean.detach(),
                f'qcmr_{stats_prefix}quality_iou_corr': correlation.detach(),
            }
            self.qcmr_debug_stats.update(stats)
        return positive_loss + self.qcmr_encoder_quality_neg_weight * negative_loss

    def loss_qcmr_quality_enc(self, outputs, targets, indices):
        loss = self._qcmr_quality_loss(outputs, targets, indices, stats_prefix='encoder_')
        return {
            'loss_qcmr_quality_enc': self.qcmr_encoder_quality_loss_weight
            * loss
        }

    def loss_qcmr_competitive_rank(self, outputs, targets, indices, epoch=0):
        """Rank Hungarian winners above nearby unmatched duplicate queries."""
        if not self.qcmr_local_rank_enabled:
            logits = outputs['pred_logits']
            return {'loss_qcmr_rank': logits.sum() * 0.0}
        if 'pred_rank_logits' not in outputs:
            raise KeyError(
                'QCCR requires training-only `pred_rank_logits` from DFINETransformer.')
        logits = outputs['pred_rank_logits']
        zero = logits.sum() * 0.0
        zero_stat = zero.detach()
        with torch.no_grad():
            rank_logit_parity = (outputs['pred_rank_logits'] - outputs['pred_logits']).abs().max()
        stats = {
            'qcmr_rank_active': logits.new_tensor(float(
                self.qcmr_local_rank_enabled and epoch >= self.qcmr_rank_start_epoch)),
            'qcmr_rank_num_pairs': zero_stat,
            'qcmr_rank_num_gt_with_pairs': zero_stat,
            'qcmr_rank_matched_pos_iou_mean': zero_stat,
            'qcmr_rank_comp_iou_mean': zero_stat,
            'qcmr_rank_pos_logit_mean': zero_stat,
            'qcmr_rank_comp_logit_mean': zero_stat,
            'qcmr_rank_score_gap_mean': zero_stat,
            'qcmr_rank_violation_rate': zero_stat,
            'qcmr_rank_better_unmatched_count': zero_stat,
            'qcmr_rank_better_unmatched_rate': zero_stat,
            'qcmr_rank_matched_winner_rate': zero_stat,
            'qcmr_rank_logit_parity_max_abs': rank_logit_parity.detach(),
        }
        self.qcmr_debug_stats.update(stats)
        if epoch < self.qcmr_rank_start_epoch:
            return {'loss_qcmr_rank': zero}

        pair_losses, pair_weights = [], []
        pos_ious, comp_ious, pos_logits, comp_logits, gaps = [], [], [], [], []
        violations = 0
        better_count = 0
        gt_with_candidates = 0
        gt_with_any_candidates = 0
        better_gt_count = 0
        winner_count = 0

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            gt_boxes = targets[batch_idx]['boxes']
            gt_labels = targets[batch_idx]['labels']
            if gt_boxes.numel() == 0 or src_idx.numel() == 0:
                continue
            pred_boxes = outputs['pred_boxes'][batch_idx].detach()
            iou_matrix, _ = box_iou(
                box_cxcywh_to_xyxy(pred_boxes),
                box_cxcywh_to_xyxy(gt_boxes))
            best_iou, best_gt = iou_matrix.max(dim=1)
            matched_mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
            matched_mask[src_idx] = True
            matched_for_gt = {int(g): int(q) for q, g in zip(src_idx.tolist(), tgt_idx.tolist())}

            for gt_idx, pos_query in matched_for_gt.items():
                iou_pos = iou_matrix[pos_query, gt_idx]
                if iou_pos < self.qcmr_rank_min_pos_iou:
                    continue
                owner_mask = best_gt == gt_idx
                candidate_mask = (~matched_mask) & owner_mask & (
                    iou_matrix[:, gt_idx] >= self.qcmr_rank_min_comp_iou)
                candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
                if candidates.numel() == 0:
                    continue
                candidate_ious = iou_matrix[candidates, gt_idx]
                gt_with_any_candidates += 1
                better_mask = candidate_ious > iou_pos + self.qcmr_rank_better_comp_tolerance
                better_count += int(better_mask.sum().item())
                better_gt_count += int(better_mask.any().item())
                candidates = candidates[~better_mask]
                candidate_ious = candidate_ious[~better_mask]
                if candidates.numel() == 0:
                    continue
                gt_with_candidates += 1
                pos_logit = logits[batch_idx, pos_query, gt_labels[gt_idx]]
                candidate_scores = logits[batch_idx, candidates, gt_labels[gt_idx]]
                order = torch.argsort(candidate_scores, descending=True)
                candidates = candidates[order[:self.qcmr_rank_topk_competitors]]
                candidate_ious = iou_matrix[candidates, gt_idx]
                candidate_scores = logits[batch_idx, candidates, gt_labels[gt_idx]]
                winner_count += int((pos_logit > candidate_scores.max()).item())
                for iou_comp, neg_logit in zip(candidate_ious, candidate_scores):
                    pair_loss = self.qcmr_rank_temperature * F.softplus(
                        (neg_logit - pos_logit) / self.qcmr_rank_temperature)
                    pair_weight = iou_pos.detach().pow(self.qcmr_rank_pos_quality_power) \
                        * iou_comp.detach().pow(self.qcmr_rank_comp_quality_power)
                    pair_losses.append(pair_loss)
                    pair_weights.append(pair_weight)
                    pos_ious.append(iou_pos.detach())
                    comp_ious.append(iou_comp.detach())
                    pos_logits.append(pos_logit.detach())
                    comp_logits.append(neg_logit.detach())
                    gaps.append((pos_logit - neg_logit).detach())
                    violations += int((neg_logit >= pos_logit).item())

        if not pair_losses:
            self.qcmr_debug_stats.update({
                'qcmr_rank_better_unmatched_count': logits.new_tensor(float(better_count)),
                'qcmr_rank_better_unmatched_rate': logits.new_tensor(
                    float(better_gt_count / max(gt_with_any_candidates, 1))),
            })
            return {'loss_qcmr_rank': zero}

        pair_losses = torch.stack(pair_losses)
        pair_weights = torch.stack(pair_weights)
        normalized_loss = (pair_weights * pair_losses).sum() / pair_weights.sum().clamp_min(self.qcmr_rank_eps)
        pair_count = len(pair_losses)
        self.qcmr_debug_stats.update({
            'qcmr_rank_num_pairs': logits.new_tensor(float(pair_count)),
            'qcmr_rank_num_gt_with_pairs': logits.new_tensor(float(gt_with_candidates)),
            'qcmr_rank_matched_pos_iou_mean': torch.stack(pos_ious).mean(),
            'qcmr_rank_comp_iou_mean': torch.stack(comp_ious).mean(),
            'qcmr_rank_pos_logit_mean': torch.stack(pos_logits).mean(),
            'qcmr_rank_comp_logit_mean': torch.stack(comp_logits).mean(),
            'qcmr_rank_score_gap_mean': torch.stack(gaps).mean(),
            'qcmr_rank_violation_rate': logits.new_tensor(float(violations / pair_count)),
            'qcmr_rank_better_unmatched_count': logits.new_tensor(float(better_count)),
            'qcmr_rank_better_unmatched_rate': logits.new_tensor(
                float(better_gt_count / max(gt_with_any_candidates, 1))),
            'qcmr_rank_matched_winner_rate': logits.new_tensor(
                float(winner_count / max(gt_with_candidates, 1))),
        })
        return {'loss_qcmr_rank': self.qcmr_rank_loss_weight * normalized_loss}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        self.qcmr_debug_stats = {}
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)['indices']
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs, ''

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {}
        for loss in self.losses:
            # TODO, indices and num_box are different from RT-DETRv2
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if self.qcmr_local_rank_enabled:
            losses.update(self.loss_qcmr_competitive_rank(
                outputs, targets, indices, epoch=kwargs.get('epoch', 0)))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.qcmr_encoder_quality_enabled and 'pred_quality' in aux_outputs:
                    quality_loss = self.loss_qcmr_quality_enc(
                        aux_outputs, enc_targets, cached_indices_enc[i])
                    # There is currently one selected-query encoder output. Keep
                    # the canonical key stable if more encoder outputs are added.
                    if i == 0:
                        losses.update(quality_loss)
                    else:
                        losses.update({f'{key}_{i}': value for key, value in quality_loss.items()})

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices


    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)


    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
