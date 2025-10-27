import numpy as np
import torch
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


def predict_masks(model, images, threshold = 0.5):
    """Runs inference on a list of preprocessed images and returns binary segmentation masks."""
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(len(images)):
            x5 = images[i].unsqueeze(0)
            logits = model(x5, target_size=x5.shape[-2:])
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (prob > threshold).astype(np.uint8)
            preds.append(pred)

    preds = np.stack(preds, axis = 0)

    return preds

def crf(image_np, prob_mask):
    """Applies dense Conditional Random Field post-processing to refine a predicted probability mask."""
    H, W = prob_mask.shape
    num_classes = 2

    probs = np.zeros((num_classes, H, W), dtype = np.float32)
    probs[1] = prob_mask
    probs[0] = 1.0 - prob_mask
    probs = probs.reshape((num_classes, -1))

    d = dcrf.DenseCRF2D(W, H, num_classes)
    d.setUnaryEnergy(unary_from_softmax(probs))
    d.addPairwiseGaussian(sxy = 5, compat = 2)
    d.addPairwiseBilateral(sxy = 60, srgb = 10, rgbim = image_np, compat = 5)

    refined = np.argmax(d.inference(5), axis = 0).reshape((H, W)).astype(np.uint8)

    return refined
