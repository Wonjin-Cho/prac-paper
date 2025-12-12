import os
import pickle
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from scipy.linalg import sqrtm

# Paths (edit if needed)
# data_path1 = "usable_synthetics/new/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle"
# label_path1 = (
#     "usable_synthetics/new/resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle"
# )

# data_path2 = "imagenet_pickle_matched/imagenet_images_matched.pickle"
# label_path2 = "imagenet_pickle_matched/imagenet_labels_matched.pickle"

# data_path3 = "image_from_pruned_model-confidence/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle"
# label_path3 = "image_from_pruned_model-confidence/resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle"

data_path1 = "usable_synthetics/new/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle"
label_path1 = (
    "usable_synthetics/new/resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle"
)

data_path2 = "imagenet_pickle_matched_batch/imagenet_images_matched.pickle"
label_path2 = "imagenet_pickle_matched_batch/imagenet_labels_matched.pickle"

data_path3 = "image_from_pruned_model-batch/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle"
label_path3 = "image_from_pruned_model-batch/resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle"

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_maybe_wrapped(path):
    with open(path, "rb") as fp:
        loaded = pickle.load(fp)
    if isinstance(loaded, (list, tuple)) and len(loaded) == 1:
        return loaded[0]
    return loaded


def ensure_images_numpy(x):
    arr = np.asarray(x)
    # If tuple/list pair (images, labels) return first element
    if arr.ndim == 0 and isinstance(x, (list, tuple)) and len(x) == 2:
        return np.asarray(x[0])
    # If shape looks like (N, ...) already return
    return arr


def normalize_range(arr):
    # convert to float32 and scale to [0,1] based on global min/max for the set
    arr = arr.astype(np.float32)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        # fallback: clip and set 0-1
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    return (arr - mn) / (mx - mn)


def prepare_images_for_inception(images_np):
    """
    Accepts images in shapes (N,C,H,W) or (N,H,W,C) or (N,H,W).
    Returns torch.Tensor of shape (N,3,299,299) normalized for Inception.
    """
    imgs = np.asarray(images_np)
    if imgs.ndim == 2:
        raise ValueError("Images appear flattened; unsupported shape.")
    # Convert channel-last to channel-first if needed
    if imgs.ndim == 4 and imgs.shape[-1] in (1, 3) and imgs.shape[1] not in (1, 3):
        imgs = np.transpose(imgs, (0, 3, 1, 2))
    if imgs.ndim == 3:
        # (N,H,W) -> (N,1,H,W)
        imgs = imgs[:, np.newaxis, :, :]
    # If single channel, repeat to 3 channels
    if imgs.shape[1] == 1:
        imgs = np.repeat(imgs, 3, axis=1)
    # scale to [0,1]
    imgs = normalize_range(imgs)
    # To torch
    imgs_t = torch.from_numpy(imgs).float()
    # resize + normalize
    transform = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Resize(299),
            T.CenterCrop(299),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # torchvision transforms expect PIL or tensor with shape (C,H,W) per sample, so apply per-batch
    batch = []
    for img in imgs_t:
        batch.append(transform(img))
    batch_t = torch.stack(batch, dim=0)
    return batch_t.to(device)


def get_activations(images_tensor, feature_extractor, batch_size=64):
    feature_extractor.eval()
    acts = []
    with torch.no_grad():
        n = images_tensor.size(0)
        for i in range(0, n, batch_size):
            batch = images_tensor[i : i + batch_size]
            out = feature_extractor(batch)
            # returned dict: pick the first value
            feat = next(iter(out.values()))
            # feat shape: (B, C, 1, 1) or (B, C)
            feat = feat.view(feat.size(0), -1).cpu().numpy()
            acts.append(feat)
    acts = np.concatenate(acts, axis=0)
    return acts


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # Sanitize means and covariances
    mu1 = np.nan_to_num(mu1, nan=0.0, posinf=0.0, neginf=0.0)
    mu2 = np.nan_to_num(mu2, nan=0.0, posinf=0.0, neginf=0.0)
    sigma1 = np.nan_to_num(sigma1, nan=0.0, posinf=0.0, neginf=0.0)
    sigma2 = np.nan_to_num(sigma2, nan=0.0, posinf=0.0, neginf=0.0)

    # Add regularization to diagonal to ensure positive semi-definite
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2
    covmean = None

    try:
        covmean = sqrtm(sigma1.dot(sigma2))
    except Exception as e:
        print(f"sqrtm failed: {e}, using identity as fallback")
        covmean = np.eye(sigma1.shape[0])

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Ensure covmean is finite
    covmean = np.nan_to_num(covmean, nan=0.0, posinf=0.0, neginf=0.0)

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(np.real(fid))


def compute_stats_from_images(images_np, feature_extractor, batch_size=64):
    imgs_t = prepare_images_for_inception(images_np)
    acts = get_activations(imgs_t, feature_extractor, batch_size=batch_size)
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def main():
    # load datasets (images only)
    a = load_maybe_wrapped(data_path1)
    b = load_maybe_wrapped(data_path2)
    c = load_maybe_wrapped(data_path3)

    imgs1 = ensure_images_numpy(a)
    imgs2 = ensure_images_numpy(b)
    imgs3 = ensure_images_numpy(c)

    print("Dataset shapes (raw):", imgs1.shape, imgs2.shape, imgs3.shape)

    # build Inceptionv3 feature extractor to get pool features (2048-d)
    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(
        device
    )
    inception.eval()

    # safer: extract 'avgpool' if present
    try:
        feat_extractor = create_feature_extractor(
            inception, return_nodes={"avgpool": "avgpool"}
        )
    except Exception:
        print("avgpool not found, trying alternative node...")
        feat_extractor = create_feature_extractor(inception, return_nodes={"fc": "fc"})

    # compute stats
    print("Computing activations and stats for dataset 1...")
    mu1, sigma1 = compute_stats_from_images(imgs1, feat_extractor)
    print("Computing activations and stats for dataset 2...")
    mu2, sigma2 = compute_stats_from_images(imgs2, feat_extractor)
    print("Computing activations and stats for dataset 3...")
    mu3, sigma3 = compute_stats_from_images(imgs3, feat_extractor)

    # compute pairwise FIDs
    fid_12 = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    fid_13 = calculate_frechet_distance(mu1, sigma1, mu3, sigma3)
    fid_23 = calculate_frechet_distance(mu2, sigma2, mu3, sigma3)

    print(f"FID (dataset1 vs dataset2): {fid_12:.4f}")
    print(f"FID (dataset1 vs dataset3): {fid_13:.4f}")
    print(f"FID (dataset2 vs dataset3): {fid_23:.4f}")


if __name__ == "__main__":
    main()
