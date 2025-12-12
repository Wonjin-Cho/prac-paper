import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import umap

from sklearn.decomposition import PCA  # optional but helpful for very high-dim input
from sklearn.impute import SimpleImputer

# Paths (edit if needed)
data_path = "image_from_pruned_model-batch/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle"
label_path = "image_from_pruned_model-batch/resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle"
# data_path = "usable_synthetics/new/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle"
# label_path = (
#     "usable_synthetics/new/resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle"
# )

data_path2 = "imagenet_pickle_matched/imagenet_images_matched.pickle"
label_path2 = "imagenet_pickle_matched/imagenet_labels_matched.pickle"


def load_maybe_wrapped(path):
    with open(path, "rb") as fp:
        loaded = pickle.load(fp)
    # If it's a single-item list/tuple like [arr], unwrap
    if isinstance(loaded, (list, tuple)) and len(loaded) == 1:
        return loaded[0]
    return loaded


def ensure_numpy_labels(x):
    arr = np.asarray(x)
    if arr.ndim > 1:
        arr = arr.flatten()
    return arr


def images_to_2d(imgs):
    """
    Convert images to 2D array (N, features).
    Supports shapes:
      - (N, C, H, W)
      - (N, H, W, C)
      - (N, D) (already flattened)
    Returns float32 array.
    """
    arr = np.asarray(imgs)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 4:
        # detect channel-first or last
        if arr.shape[1] in (1, 3):
            N, C, H, W = arr.shape
            return arr.reshape(N, C * H * W).astype(np.float32)
        else:
            # maybe channel-last
            N, H, W, C = arr.shape
            return arr.reshape(N, H * W * C).astype(np.float32)
    if arr.ndim == 3:
        # ambiguous: assume (N, H, W) -> treat as single channel
        N, H, W = arr.shape
        return arr.reshape(N, H * W).astype(np.float32)
    raise ValueError(f"Unsupported image array shape: {arr.shape}")


# --- Load datasets ---
print("Loading synthetic dataset...")
syn_data_raw = load_maybe_wrapped(data_path)
syn_labels_raw = load_maybe_wrapped(label_path)

print("Loading real (ImageNet matched) dataset...")
real_data_raw = load_maybe_wrapped(data_path2)
real_labels_raw = load_maybe_wrapped(label_path2)

# Some pickles may store (images, labels) tuple — handle that
if (
    isinstance(syn_data_raw, (list, tuple))
    and len(syn_data_raw) == 2
    and isinstance(syn_data_raw[0], np.ndarray)
):
    syn_data_raw = syn_data_raw[0]
if (
    isinstance(syn_labels_raw, (list, tuple))
    and len(syn_labels_raw) == 2
    and isinstance(syn_labels_raw[1], np.ndarray)
):
    syn_labels_raw = syn_labels_raw[1]

if (
    isinstance(real_data_raw, (list, tuple))
    and len(real_data_raw) == 2
    and isinstance(real_data_raw[0], np.ndarray)
):
    real_data_raw = real_data_raw[0]
if (
    isinstance(real_labels_raw, (list, tuple))
    and len(real_labels_raw) == 2
    and isinstance(real_labels_raw[1], np.ndarray)
):
    real_labels_raw = real_labels_raw[1]

# Ensure numpy arrays and 1D labels
syn_labels = ensure_numpy_labels(syn_labels_raw)
real_labels = ensure_numpy_labels(real_labels_raw)

# Convert images to 2D feature arrays
syn_feats = images_to_2d(syn_data_raw)
real_feats = images_to_2d(real_data_raw)

print("Synthetic:", syn_feats.shape, "labels:", syn_labels.shape)
print("Real    :", real_feats.shape, "labels:", real_labels.shape)

# --- Combine and reduce dimensionality for UMAP ---
X = np.concatenate([syn_feats, real_feats], axis=0)
y = np.concatenate([syn_labels, real_labels], axis=0)
source = np.concatenate(
    [np.zeros(len(syn_feats), dtype=int), np.ones(len(real_feats), dtype=int)], axis=0
)
print("Combined shape:", X.shape)

# Optional PCA to speed up UMAP on very high-dim data
pca_components = 50

# Impute NaN/inf values before PCA/UMAP
if not np.isfinite(X).all():
    print("Found NaN/inf in X — imputing missing values (column mean)")
    # mark non-finite as NaN so SimpleImputer can handle them
    X[~np.isfinite(X)] = np.nan
    try:
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)
    except Exception as e:
        print("SimpleImputer failed:", e, "- filling NaN/inf with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

if X.shape[1] > 500 and pca_components < X.shape[1]:
    print("Running PCA ->", pca_components)
    X = PCA(n_components=min(pca_components, X.shape[1])).fit_transform(X)

# Run UMAP
print("Running UMAP...")
umap_model = umap.UMAP(
    n_components=2, n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42
)
emb = umap_model.fit_transform(X)
print("UMAP done:", emb.shape)

# --- Plotting ---
plt.figure(figsize=(12, 6))

# Color by dataset source only (ignore class labels)
labels_names = {0: "Synthetic", 1: "Real (matched ImageNet)"}
cmap = plt.get_cmap("Set1")
source_colors = {0: cmap(0), 1: cmap(1)}
markers = {0: "o", 1: "s"}

for src in (0, 1):
    mask = source == src
    plt.scatter(
        emb[mask, 0],
        emb[mask, 1],
        c=[source_colors[src]] * mask.sum(),
        s=10,
        marker=markers[src],
        edgecolors="none",
        alpha=0.9,
        label=labels_names[src],
    )

plt.legend(title="Dataset", loc="upper right")
plt.title("UMAP: Synthetic vs Real — colored by dataset source")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()

out_path = "umap_compare_by_source.png"
plt.savefig(out_path, dpi=300)
plt.close()
print("Saved UMAP comparison to", out_path)
