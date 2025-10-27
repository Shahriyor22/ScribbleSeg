import torch


def focal_loss(pred, target, mask, alpha = 0.25, gamma = 2, eps = 1e-6):
    """Calculates the Focal loss between predicted logits and target masks."""
    pred = torch.clamp(torch.sigmoid(pred), eps, 1 - eps) * mask
    target *= mask

    p = torch.where(target == 1, pred, 1 - pred)
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)

    focal = -alpha_t * (1 - p) ** gamma * torch.log(p)

    return focal.mean()

def dice_loss(pred, target, mask):
    """Calculates the Dice loss between predicted logits and target masks."""
    pred = torch.sigmoid(pred) * mask
    target *= mask

    intersection = (pred * target).sum(dim = (1, 2, 3))
    pred_sum = pred.sum(dim = (1, 2, 3))
    target_sum = target.sum(dim = (1, 2, 3))

    dice = (2 * intersection) / (pred_sum + target_sum)

    return 1 - dice.mean()

def combined_loss(pred, target, mask):
    """Calculates the sum of Focal and Dice losses for balanced segmentation supervision."""
    return focal_loss(pred, target, mask) + dice_loss(pred, target, mask)

def hybrid_loss(pred, target, mask, gt, lam = 0.3):
    """Combines scribble-supervised and ground-truth losses to balance learning."""
    return lam * combined_loss(pred, target, mask) + (1 - lam) * combined_loss(pred, gt, torch.ones_like(gt))
