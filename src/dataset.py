import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image


def loader(tensors, batch_size = 4, shuffle = False):
    """Creates a DataLoader from given tensors with safe default settings (no multiprocessing)."""
    return DataLoader(TensorDataset(*tensors), batch_size = batch_size, shuffle = shuffle, num_workers = 0,
                      pin_memory = False, persistent_workers = False)

def augment_batch(image, label, mask, p_flip = 0.5, p_jitter = 0.5, jitter_brightness = 0.15, jitter_contrast = 0.15,
                  p_noise = 0.3, noise_std = 0.02, p_drop_scrib = 0.3):
    """Applies data augmentations to a batch of 5-channel inputs (RGB + 2 scribble maps)."""
    B, C, H, W = image.shape

    if C < 3:
        raise RuntimeError(f"Expected at least 3 channels (RGB), got {C}.")

    if torch.rand(1).item() < p_flip:
        image = torch.flip(image, dims = [-1])
        label = torch.flip(label, dims = [-1])
        mask = torch.flip(mask, dims = [-1])

    if torch.rand(1).item() < p_jitter:
        rgb = image[:, :3]
        mean = rgb.mean(dim = (2,3), keepdim = True)
        c = 1.0 + (2 * torch.rand(1).item() - 1.0) * jitter_contrast
        rgb = (rgb - mean) * c + mean
        b = (2 * torch.rand(1).item() - 1.0) * jitter_brightness
        rgb = torch.clamp(rgb + b, 0.0, 1.0)
        image[:, :3] = rgb

    if torch.rand(1).item() < p_noise:
        noise = torch.randn_like(image[:, :3]) * noise_std
        image[:, :3] = torch.clamp(image[:, :3] + noise, 0.0, 1.0)

    if image.shape[1] >= 5 and torch.rand(1).item() < p_drop_scrib:
        image[:, 3:5] = 0

    return image, label, mask

def store_predictions(predictions, filenames):
    """Takes a stack of segmented images and stores them individually in the given folder."""
    pred_dir = "data/predictions"
    gtr_dir = "data/ground_truths"

    palette = Image.open(os.path.join(gtr_dir, filenames[0] + ".png")).getpalette()

    for stem, mask in zip(filenames, predictions):
        save_path = os.path.join(pred_dir, f"{stem}.png")
        img = Image.fromarray(mask.astype(np.uint8), mode = "P")
        img.putpalette(palette)
        img.save(save_path)
