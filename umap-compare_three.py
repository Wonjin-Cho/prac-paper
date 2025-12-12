import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Paths (edit if needed)
data_path1 = "usable_synthetics/new/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle"
label_path1 = (
    "usable_synthetics/new/resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle"
)

data_path2 = "imagenet_pickle_matched_batch/imagenet_images_matched.pickle"
label_path2 = "imagenet_pickle_matched_batch/imagenet_labels_matched.pickle"

data_path3 = "image_from_pruned_model-batch/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle"
label_path3 = "image_from_pruned_model-batch/resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle"


def load_maybe_wrapped(path):
    with open(path, "rb") as fp:
        loaded = pickle.load(fp)
    if isinstance(loaded, (list, tuple)) and len(loaded) == 1:
        return loaded[0]
    return loaded


def ensure_numpy_labels(x):
    arr = np.asarray(x)
    if arr.ndim > 1:
        arr = arr.flatten()
    return arr


def images_to_2d(imgs):
    arr = np.asarray(imgs)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 4:
        if arr.shape[1] in (1, 3):
            N, C, H, W = arr.shape
            return arr.reshape(N, C * H * W).astype(np.float32)
        else:
            N, H, W, C = arr.shape
            return arr.reshape(N, H * W * C).astype(np.float32)
    if arr.ndim == 3:
        N, H, W = arr.shape
        return arr.reshape(N, H * W).astype(np.float32)
    raise ValueError(f"Unsupported image array shape: {arr.shape}")


# --- Load datasets ---
print("Loading dataset 1...")
data1_raw = load_maybe_wrapped(data_path1)
labels1_raw = load_maybe_wrapped(label_path1)

print("Loading dataset 2...")
data2_raw = load_maybe_wrapped(data_path2)
labels2_raw = load_maybe_wrapped(label_path2)

print("Loading dataset 3...")
data3_raw = load_maybe_wrapped(data_path3)
labels3_raw = load_maybe_wrapped(label_path3)

# Handle potential tuple unpacking
for data_raw, labels_raw in [
    (data1_raw, labels1_raw),
    (data2_raw, labels2_raw),
    (data3_raw, labels3_raw),
]:
    if (
        isinstance(data_raw, (list, tuple))
        and len(data_raw) == 2
        and isinstance(data_raw[0], np.ndarray)
    ):
        data_raw = data_raw[0]
    if (
        isinstance(labels_raw, (list, tuple))
        and len(labels_raw) == 2
        and isinstance(labels_raw[1], np.ndarray)
    ):
        labels_raw = labels_raw[1]

# Ensure numpy arrays and 1D labels
labels1 = ensure_numpy_labels(labels1_raw)
labels2 = ensure_numpy_labels(labels2_raw)
labels3 = ensure_numpy_labels(labels3_raw)

# Convert images to 2D feature arrays
feats1 = images_to_2d(data1_raw)
feats2 = images_to_2d(data2_raw)
feats3 = images_to_2d(data3_raw)

print("Dataset 1:", feats1.shape, "labels:", labels1.shape)
print("Dataset 2:", feats2.shape, "labels:", labels2.shape)
print("Dataset 3:", feats3.shape, "labels:", labels3.shape)

# --- Combine and reduce dimensionality for UMAP ---
X = np.concatenate([feats1, feats2, feats3], axis=0)
y = np.concatenate([labels1, labels2, labels3], axis=0)
source = np.concatenate(
    [
        np.zeros(len(feats1), dtype=int),
        np.ones(len(feats2), dtype=int),
        np.full(len(feats3), 2, dtype=int),
    ],
    axis=0,
)
print("Combined shape:", X.shape)

# Optional PCA to speed up UMAP on very high-dim data
pca_components = 50

# Impute NaN/inf values before PCA/UMAP
if not np.isfinite(X).all():
    print("Found NaN/inf in X — imputing missing values (column mean)")
    X[~np.isfinite(X)] = np.nan
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

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

# Plot by dataset source (color = dataset, marker = dataset)
markers = {
    0: "o",
    1: "s",
    2: "D",
}  # 0: dataset1 (circle), 1: dataset2 (square), 2: dataset3 (diamond)
labels_names = {
    0: "Synthetic images from full model",
    1: "ImageNet",
    2: "Synthetic images from compressed model",
}

# Choose distinct colors for the three datasets
cmap = plt.get_cmap("Set1")
source_colors = {0: cmap(0), 1: cmap(1), 2: cmap(2)}

for src in (0, 1, 2):
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

# Legend for source types
plt.legend(title="Dataset", loc="upper right")

plt.title("UMAP: Comparison of Three Datasets — colored by dataset source")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()

out_path = "umap_compare_three_datasets_2.png"
plt.savefig(out_path, dpi=300)
plt.close()
print("Saved UMAP comparison to", out_path)
