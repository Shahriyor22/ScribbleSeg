import numpy as np
import matplotlib.pyplot as plt


def overlay_scribbles(image, scribble, color_fg = (255, 0, 0), color_bg = (0, 0, 255), alpha = 0.6):
    """Overlays foreground and background scribbles on an RGB image"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be RGB.")

    if scribble.shape != image.shape[:2]:
        raise ValueError("Scribble must match image spatial size.")

    overlaid = image.copy().astype(np.float32)

    mask_fg = (scribble == 1)
    mask_bg = (scribble == 0)

    for mask, color in [(mask_fg, color_fg), (mask_bg, color_bg)]:
        for c in range(3):
            overlaid[..., c][mask] = alpha * color[c] + (1 - alpha) * overlaid[..., c][mask]

    return overlaid.astype(np.uint8)

def visualize(image, scribble, ground_truth, prediction, alpha = 0.6):
    """Shows a subplot of an image overlaid with scribbles, ground-truth and prediction masks (blue = bg, red = fg)."""
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))

    axes[0].imshow(overlay_scribbles(image, scribble, alpha = alpha))
    axes[0].set_title("Image + Scribble")

    axes[1].imshow(ground_truth, cmap = plt.get_cmap('bwr'), vmin = 0, vmax = 1)
    axes[1].set_title("Ground-truth")

    axes[2].imshow(prediction, cmap = plt.get_cmap('bwr'), vmin = 0, vmax = 1)
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    plt.show()
