"""DN-DETR-style query denoising components for DEIM."""

import torch
import torch.nn.functional as F
import torchvision

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .utils import inverse_sigmoid


def prepare_for_dn(targets, class_embed, batch_size, training, num_queries,
                   num_classes, hidden_dim, dn_number=50,
                   label_noise_ratio=0.5, box_noise_scale=0.2):
    """Build non-contrastive noisy GT queries and their attention mask."""
    if not training or dn_number <= 0:
        return None, None, None, None
    if targets is None or len(targets) != batch_size:
        raise ValueError('targets must contain one item per training image.')
    if not 0.0 <= label_noise_ratio <= 1.0:
        raise ValueError('label_noise_ratio must be in [0, 1].')
    if box_noise_scale < 0:
        raise ValueError('box_noise_scale must be non-negative.')
    if class_embed.embedding_dim != hidden_dim:
        raise ValueError('class embedding dimension must equal hidden_dim.')

    num_gts = [len(target['labels']) for target in targets]
    max_gt = max(num_gts, default=0)
    if max_gt == 0:
        return None, None, None, None

    device = targets[0]['labels'].device
    dtype = targets[0]['boxes'].dtype
    num_groups = max(int(dn_number) // max_gt, 1)
    dn_count = max_gt * num_groups

    input_labels = torch.full(
        (batch_size, max_gt), num_classes, dtype=torch.long, device=device)
    input_boxes = torch.zeros(
        (batch_size, max_gt, 4), dtype=dtype, device=device)
    valid_mask = torch.zeros(
        (batch_size, max_gt), dtype=torch.bool, device=device)
    positive_indices = []

    for batch_idx, target in enumerate(targets):
        num_gt = num_gts[batch_idx]
        if num_gt:
            input_labels[batch_idx, :num_gt] = target['labels']
            input_boxes[batch_idx, :num_gt] = target['boxes']
            valid_mask[batch_idx, :num_gt] = True
            group_offsets = torch.arange(num_groups, device=device) * max_gt
            gt_indices = torch.arange(num_gt, device=device)
            positive_indices.append((group_offsets[:, None] + gt_indices).reshape(-1))
        else:
            positive_indices.append(torch.empty(0, dtype=torch.long, device=device))

    input_labels = input_labels.tile(1, num_groups)
    input_boxes = input_boxes.tile(1, num_groups, 1)
    valid_mask = valid_mask.tile(1, num_groups)

    if label_noise_ratio > 0:
        noise_mask = torch.rand(input_labels.shape, device=device) < label_noise_ratio
        random_labels = torch.randint(
            0, num_classes, input_labels.shape, dtype=torch.long, device=device)
        input_labels = torch.where(noise_mask & valid_mask, random_labels, input_labels)

    if box_noise_scale > 0:
        box_delta = torch.zeros_like(input_boxes)
        box_delta[..., :2] = input_boxes[..., 2:] * 0.5
        box_delta[..., 2:] = input_boxes[..., 2:]
        box_noise = (torch.rand_like(input_boxes) * 2.0 - 1.0) * box_delta
        noisy_boxes = (input_boxes + box_noise * box_noise_scale).clamp(0.0, 1.0)
        input_boxes = torch.where(valid_mask[..., None], noisy_boxes, input_boxes)

    input_query_logits = class_embed(input_labels)
    input_query_bbox_unact = inverse_sigmoid(input_boxes)

    total_queries = dn_count + num_queries
    attn_mask = torch.zeros(
        (total_queries, total_queries), dtype=torch.bool, device=device)
    # Matching queries cannot use the noisy GT queries as shortcuts.
    attn_mask[dn_count:, :dn_count] = True
    # Each DN reconstruction group is independent from every other DN group.
    for group_idx in range(num_groups):
        start = group_idx * max_gt
        end = start + max_gt
        attn_mask[start:end, :start] = True
        attn_mask[start:end, end:dn_count] = True

    dn_meta = {
        'dn_positive_idx': tuple(positive_indices),
        'dn_num_group': num_groups,
        'dn_num_split': [dn_count, num_queries],
    }
    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta):
    """Split DN predictions from normal predictions along the query axis."""
    if dn_meta is None:
        return None, outputs_class, None, outputs_coord
    dn_count, normal_count = dn_meta['dn_num_split']
    dn_class, normal_class = torch.split(
        outputs_class, [dn_count, normal_count], dim=-2)
    dn_coord, normal_coord = torch.split(
        outputs_coord, [dn_count, normal_count], dim=-2)
    return dn_class, normal_class, dn_coord, normal_coord


def compute_dn_loss(outputs, targets, dn_meta, num_classes, num_boxes,
                    alpha=0.25, gamma=2.0, loss_weight=1.0):
    """Compute classification, L1, and GIoU losses for known DN queries."""
    logits = outputs['pred_logits']
    boxes = outputs['pred_boxes']
    zero = logits.sum() * 0.0
    positive_idx = dn_meta.get('dn_positive_idx', ()) if dn_meta else ()
    num_groups = int(dn_meta.get('dn_num_group', 1)) if dn_meta else 1
    batch_indices, query_indices, target_labels, target_boxes = [], [], [], []

    for batch_idx, (target, query_idx) in enumerate(zip(targets, positive_idx)):
        if query_idx.numel() == 0:
            continue
        labels = target['labels'].repeat(num_groups)
        boxes_target = target['boxes'].repeat((num_groups, 1))
        if labels.numel() != query_idx.numel():
            raise ValueError('DN positive-query count does not match repeated targets.')
        batch_indices.append(torch.full_like(query_idx, batch_idx))
        query_indices.append(query_idx)
        target_labels.append(labels)
        target_boxes.append(boxes_target)

    if not query_indices:
        return {'loss_dn_cls': zero, 'loss_dn_bbox': zero, 'loss_dn_giou': zero}

    selected = (torch.cat(batch_indices), torch.cat(query_indices))
    target_labels = torch.cat(target_labels)
    target_boxes = torch.cat(target_boxes)
    src_logits = logits[selected]
    target_onehot = F.one_hot(
        target_labels, num_classes=num_classes).to(src_logits.dtype)
    loss_cls = torchvision.ops.sigmoid_focal_loss(
        src_logits, target_onehot, alpha, gamma, reduction='none').sum()

    src_boxes = boxes[selected]
    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='sum')
    loss_giou = 1 - torch.diag(generalized_box_iou(
        box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
    normalizer = max(float(num_boxes), 1.0)
    return {
        'loss_dn_cls': loss_weight * loss_cls / normalizer,
        'loss_dn_bbox': loss_weight * loss_bbox / normalizer,
        'loss_dn_giou': loss_weight * loss_giou.sum() / normalizer,
    }
