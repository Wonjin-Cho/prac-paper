import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import Counter


def create_imagenet_pickle_matched(
    data_path="usable_synthetics/new/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle",
    label_path="usable_synthetics/new/resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle",
    output_dir="./imagenet_pickle_matched_original",
):
    """
    Load reference pickle dataset, extract label distribution, then sample from ImageNet
    matching the same label distribution.

    Args:
        data_path: Path to reference pickle images
        label_path: Path to reference pickle labels
        output_dir: Directory to save matched pickle files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load reference pickle dataset
    print("Loading reference pickle dataset...")
    with open(label_path, "rb") as f:
        [ref_labels] = pickle.load(f)

    # Flatten if multi-dimensional
    if isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.flatten()

    # Convert to list for Counter compatibility
    ref_labels = (
        ref_labels.tolist() if isinstance(ref_labels, np.ndarray) else list(ref_labels)
    )

    # Count label distribution
    label_counts = Counter(ref_labels)
    total_samples = len(ref_labels)
    print(f"Reference label distribution: {dict(label_counts)}")
    print(f"Total samples to match: {total_samples}")

    # ImageNet normalization and preprocessing
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Load ImageNet validation set
    imagenet_path = "../../ImageNet/ILSVRC2012_img_train"  # Adjust path if needed
    if not os.path.exists(imagenet_path):
        print(f"Error: ImageNet dataset not found at {imagenet_path}")
        return

    imagenet_dataset = datasets.ImageFolder(
        os.path.join(imagenet_path, "val"), transform=transform
    )

    # Create indices for each class in ImageNet
    print("Building ImageNet class indices...")
    class_indices = {}
    for idx, (_, label) in enumerate(imagenet_dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Sample indices matching the label distribution
    selected_indices = []
    for label, count in label_counts.items():
        if label not in class_indices:
            print(
                f"Warning: Label {label} not found in ImageNet, skipping {count} samples"
            )
            continue

        available_indices = class_indices[label]
        if len(available_indices) < count:
            print(
                f"Warning: Only {len(available_indices)} samples available for label {label}, need {count}"
            )
            sampled = available_indices
        else:
            sampled = np.random.choice(available_indices, size=count, replace=False)

        selected_indices.extend(sampled)
        print(f"Sampled {len(sampled)} images for label {label}")

    # Create subset dataset
    subset = Subset(imagenet_dataset, selected_indices)
    dataloader = DataLoader(subset, batch_size=64, shuffle=False)

    # Collect sampled images and labels
    all_images = []
    all_labels = []

    print(f"Collecting matched ImageNet samples...")
    for batch_idx, (images, labels) in enumerate(dataloader):
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * 64} images...")

    # Concatenate all batches
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"Total images collected: {all_images.shape}")
    print(f"Total labels collected: {all_labels.shape}")
    print(f"Matched label distribution: {dict(Counter(all_labels))}")

    # Save as pickle files
    images_path = os.path.join(output_dir, "imagenet_images_matched.pickle")
    labels_path = os.path.join(output_dir, "imagenet_labels_matched.pickle")

    with open(images_path, "wb") as f:
        pickle.dump(all_images, f)
    print(f"Saved images to {images_path}")

    with open(labels_path, "wb") as f:
        pickle.dump(all_labels, f)
    print(f"Saved labels to {labels_path}")

    print(f"✓ Matched ImageNet pickle dataset created successfully!")
    return all_images, all_labels


if __name__ == "__main__":
    create_imagenet_pickle_matched(
        data_path="image_from_pruned_model-batch/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle",
        label_path="image_from_pruned_model-batch/resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle",
        output_dir="./imagenet_pickle_matched_batch",
    )
