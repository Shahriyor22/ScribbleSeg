import os
import numpy as np
import torch
from sklearn.semi_supervised import LabelSpreading
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import slic
from PIL import Image
from tqdm import tqdm


def spectral_label_propagation(image_np, scribble_np, segments, gamma = 20.0, max_iter = 50, conf_thr = 0.6):
    """Propagates sparse scribble labels to all superpixels using LabelSpreading (RBF kernel)."""
    H, W, _ = image_np.shape
    yy, xx = np.mgrid[0:H, 0:W]

    num_segments = segments.max() + 1
    feats, labels = [], []

    for sid in range(num_segments):
        m = (segments == sid)
        rgb = image_np[m].mean(0) / 255
        x = xx[m].mean() / W
        y = yy[m].mean() / H
        feats.append(np.hstack([rgb, x, y]))

        s = scribble_np[m]
        s = s[s != 255]
        labels.append(int(s[0]) if len(s) else -1)

    X = np.stack(feats).astype(np.float32)
    y = np.array(labels, dtype = int)

    labeled = y[y != -1]
    uniq = np.unique(labeled) if labeled.size > 0 else np.array([])

    if uniq.size < 2:
        seeded_fg = (y == 1)
        seeded_bg = (y == 0)

        p_fg_sp = np.full(num_segments, 0.5, dtype = np.float32)
        p_fg_sp[seeded_fg] = 1.0
        p_fg_sp[seeded_bg] = 0.0

        conf_sp = np.zeros(num_segments, dtype = np.float32)
        conf_sp[seeded_fg | seeded_bg] = 1.0

    else:
        lp = LabelSpreading(kernel = 'rbf', gamma = gamma, max_iter = max_iter)
        lp.fit(X, y)

        probs = lp.label_distributions_
        classes = lp.classes_

        if (classes == 1).any():
            idx_fg = int(np.where(classes == 1)[0][0])
            p_fg_sp = probs[:, idx_fg].astype(np.float32)
        else:
            p_fg_sp = np.zeros(num_segments, dtype = np.float32)

        conf_sp = np.abs(p_fg_sp - 0.5) * 2.0

    prob = np.zeros((H, W), np.float32)
    conf = np.zeros((H, W), np.float32)

    for sid in range(num_segments):
        m = (segments == sid)
        prob[m] = p_fg_sp[sid]
        conf[m] = conf_sp[sid]

    if conf_thr is None:
        mask_conf = (conf > 0).astype(np.uint8)
    else:
        mask_conf = (conf >= float(conf_thr)).astype(np.uint8)

    propagated = (prob >= 0.5).astype(np.uint8)

    return propagated, mask_conf

def build_wmap(conf_map, scribble_np, tau = 60):
    """Builds a per-pixel training weight map emphasizing close-to-scribble regions."""
    fg = (scribble_np == 1)
    seeded = (scribble_np != 255).astype(np.float32)

    d_fg = distance_transform_edt(~fg)
    prox = np.exp(-(d_fg / float(tau)))

    w = np.maximum(seeded, conf_map * prox).astype(np.float32)

    return w

def scribble_channels(scribble_np, sigma = 40.0):
    """Creates two distance-based scribble guidance channels (fg/bg)."""
    fg = (scribble_np == 1)
    bg = (scribble_np == 0)

    d_fg = distance_transform_edt(~fg)
    d_bg = distance_transform_edt(~bg)

    c_fg = np.exp(-d_fg / sigma).astype(np.float32)
    c_bg = np.exp(-d_bg / sigma).astype(np.float32)

    return np.stack([c_fg, c_bg], 0)

def preprocess(image, scribble):
    """End-to-end preprocessing to build model inputs and training targets from an image + scribbles."""
    image_np = np.array(image)
    scribble_np = np.array(scribble)

    segments = slic(image_np, n_segments = 2500, compactness = 12, start_label = 0)
    prob_fg, conf = spectral_label_propagation(image_np, scribble_np, segments, gamma = 8.0, max_iter = 50)

    weight = build_wmap(conf, scribble_np, tau = 150)
    y_soft = prob_fg.astype(np.float32)[None, ...]
    w_map = weight[None, ...].astype(np.float32)

    rgb = image_np.transpose(2, 0, 1).astype(np.float32) / 255.0
    extra = scribble_channels(scribble_np, sigma = 100.0)
    x5 = np.concatenate([rgb, extra], axis = 0)

    img_t = torch.tensor(x5)
    lbl_t = torch.from_numpy(y_soft)
    mask_t = torch.from_numpy(w_map)

    return img_t, lbl_t, mask_t

if __name__ == "__main__":
    def by_stem(folder, extension):
        """Creates a mapping from filename stems to full paths for files with the given extension."""
        files = [f for f in os.listdir(folder) if f.lower().endswith(tuple(extension)) and not f.startswith(".")]

        return {os.path.splitext(f)[0]: os.path.join(folder, f) for f in files}

    img_dir = "data/images"
    scr_dir = "data/scribbles"
    gtr_dir = "data/ground_truths"

    img_map = by_stem(img_dir, (".jpg",))
    scr_map = by_stem(scr_dir, (".png",))
    gtr_map = by_stem(gtr_dir, (".png",))

    stems = sorted(set(img_map) & set(scr_map) & set(gtr_map))

    if not stems:
        raise RuntimeError(f"No matching triplets in {img_dir}, {scr_dir} and {gtr_dir}.")

    imgs, lbls, msks, gtrs, fnames = [], [], [], [], []

    for stem in tqdm(stems, desc = "Preprocessing"):
        img = Image.open(img_map[stem]).convert("RGB")
        scr = Image.open(scr_map[stem]).convert("L")

        img_t, lbl_t, mask_t = preprocess(img, scr)
        gt_t = torch.from_numpy((np.array(Image.open(gtr_map[stem])) == 1).astype(np.float32)[None, ...])

        imgs.append(img_t)
        lbls.append(lbl_t)
        msks.append(mask_t)
        gtrs.append(gt_t)
        fnames.append(stem)

    imgs = torch.stack(imgs)
    lbls = torch.stack(lbls)
    msks = torch.stack(msks)
    gtrs = torch.stack(gtrs)

    save_path = "data/preprocessed/preprocessed.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    torch.save({"images": imgs, "labels": lbls, "masks": msks, "ground_truths": gtrs, "filenames": fnames},
               save_path)

    print(f"Saved {imgs.shape}, {lbls.shape}, {msks.shape}, {gtrs.shape} -> {save_path}")
