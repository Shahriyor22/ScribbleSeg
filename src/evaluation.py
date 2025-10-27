import numpy as np
import torch


def miou(gts, preds):
    """Computes the average IoU for foreground, background and their mean across all samples."""
    ious_fg, ious_bg, ious_mean = [], [], []

    for gt, pred in zip(gts, preds):
        gt = np.array(gt).astype(np.uint8)
        pred = pred.astype(np.uint8)

        # fg (1)
        inter_fg = np.logical_and(gt == 1, pred == 1).sum()
        union_fg = np.logical_or(gt == 1, pred == 1).sum()
        iou_fg = 1.0 if union_fg == 0 else inter_fg / union_fg

        # bg (0)
        inter_bg = np.logical_and(gt == 0, pred == 0).sum()
        union_bg = np.logical_or(gt == 0, pred == 0).sum()
        iou_bg = 1.0 if union_bg == 0 else inter_bg / union_bg

        # mean
        iou_mean = (iou_fg + iou_bg) / 2

        ious_fg.append(iou_fg)
        ious_bg.append(iou_bg)
        ious_mean.append(iou_mean)

    print(f"FG IoU: {np.mean(ious_fg):.4f}, BG IoU: {np.mean(ious_bg):.4f}, Mean IoU: {np.mean(ious_mean):.4f}")

def miou_batch(logits, gt, eps = 1e-9):
    """Computes batch-wise mean IoU metrics from logits and ground-truths."""
    pred = (torch.sigmoid(logits) > 0.5).float()

    # fg (1)
    inter_fg = (pred * gt).sum(dim = (1, 2, 3))
    union_fg = ((pred + gt) > 0).sum(dim = (1, 2, 3))
    iou_fg = inter_fg / (union_fg + eps)

    # bg (0)
    inter_bg = ((1 - pred) * (1 - gt)).sum(dim = (1, 2, 3))
    union_bg = (((1 - pred) + (1 - gt)) > 0).sum(dim = (1, 2, 3))
    iou_bg = inter_bg / (union_bg + eps)

    # mean
    iou_mean = (iou_fg + iou_bg) / 2

    return {"fg": iou_fg.mean().item(), "bg": iou_bg.mean().item(), "mean": iou_mean.mean().item()}
