import torch

def photometric_loss(pred, target, mask):
    """Photometric loss between predicted and target images.
    Args:
        pred: predicted image, (B, 3, H, W)
        target: target image, (B, 3, H, W)
        mask: mask for valid pixels, (B, 1, H, W)
    Returns:
        photometric loss, (B)
    """
    diff = (pred - target) * mask
    return torch.mean(diff ** 2, dim=[1, 2, 3])