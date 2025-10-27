import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from PIL import Image

from src.dataset import loader, store_predictions
from src.model import UNetResNet34
from src.loss import hybrid_loss
from src.segmentation import predict_masks, crf
from src.plot import visualize
from src.evaluation import miou, miou_batch


# Preprocessed pack loading
pack = torch.load("data/preprocessed/preprocessed.pt", map_location = "cpu")

imgs = pack["images"].float()
lbls = pack["labels"].float()
msks = pack["masks"].float()
gtrs = pack["ground_truths"].float()
fnames = pack["filenames"]

# Index splitting
idx = np.arange(imgs.shape[0])
train_idx, valid_idx = train_test_split(idx, test_size = 0.2, random_state = 42, shuffle = True)

train_loader = loader((imgs[train_idx], lbls[train_idx], msks[train_idx], gtrs[train_idx]),
                      batch_size = 2, shuffle = True)
valid_loader = loader((imgs[valid_idx], lbls[valid_idx], msks[valid_idx], gtrs[valid_idx]),
                      batch_size = 2, shuffle = False)

# Model definition
model = UNetResNet34()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-4)

# Hyperparameters
epochs = 20
epochs_no_improve = 0
patience = 5
best_valid_miou = 0.0
best_model_path = "best_model.pth"

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for ib, lb, mb, gb in train_loader:
        optimizer.zero_grad()
        logits = model(ib)
        loss = hybrid_loss(logits, lb, mb, gb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    valid_loss, valid_miou, batches = 0.0, 0.0, 0

    with torch.no_grad():
        for ib, lb, mb, gb in valid_loader:
            logits = model(ib)
            valid_loss += hybrid_loss(logits, lb, mb, gb).item()
            valid_iou = miou_batch(logits, gb)
            valid_miou += valid_iou["mean"]
            batches += 1

    train_loss /= max(1, len(train_loader))
    valid_loss /= max(1, batches)
    valid_miou /= max(1, batches)

    print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f}, "
          f"Valid Loss = {valid_loss:.4f}, Valid Mean IoU = {valid_miou:.4f}")

    # Early stopping
    if valid_miou > best_valid_miou:
        best_valid_miou, epochs_no_improve = valid_miou, 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement...")
            break

# Best model loading
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Inference: creating an array of size num_valid x 375 x 500, a stack of all the segmented images (1 = fg, 0 = bg)
valid_idx_list = valid_idx.tolist() if isinstance(valid_idx, np.ndarray) else list(valid_idx)

gtrs_valid = gtrs[valid_idx_list]
fnames_valid = [fnames[i] for i in valid_idx_list]
preds_valid = predict_masks(model, imgs[valid_idx_list])

# CRF post-Processing
preds_refined = []

for i, fname in enumerate(fnames_valid):
    image_np = np.array(Image.open(os.path.join("data/images", fname + ".jpg")).convert("RGB"))
    prob_mask = preds_valid[i].astype(np.float32)
    refined_mask = crf(image_np, prob_mask)
    preds_refined.append(refined_mask)

preds_refined = np.stack(preds_refined, axis = 0)

# Performance measurement
miou(gtrs_valid, preds_refined)

# Performance visualization
ex_idx = np.random.randint(len(valid_idx_list))

img_ex = np.array(Image.open(os.path.join("data/images", fnames_valid[ex_idx] + ".jpg")).convert("RGB"))
scr_ex = np.array(Image.open(os.path.join("data/scribbles", fnames_valid[ex_idx] + ".png")).convert("L"))
gtr_ex = gtrs_valid[ex_idx].squeeze(0).numpy().astype(np.uint8)
pred_ex = preds_refined[ex_idx].astype(np.uint8)

visualize(img_ex, scr_ex, gtr_ex, pred_ex)

# Predictions storage
preds = predict_masks(model, imgs)
store_predictions(preds, fnames)
