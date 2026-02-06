import numpy as np
import cv2
from sklearn.decomposition import PCA


def _infer_grid_size(num_patches, image_size, patch_size):
    if image_size is not None and patch_size is not None:
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        grid = int(image_size // patch_size)
        if grid * grid == num_patches:
            return (grid, grid)
    grid = int(np.sqrt(num_patches))
    if grid * grid != num_patches:
        raise ValueError(f"Cannot infer square grid from num_patches={num_patches}.")
    return (grid, grid)


def compute_foreground_mask(
    patch_tokens,
    image_size,
    patch_size,
    threshold=10.0,
    kernel_size=3,
    border=0.2,
    min_center_ratio=0.35,
):
    if patch_tokens.ndim != 2:
        raise ValueError("patch_tokens must be 2D [N, D].")
    num_patches = patch_tokens.shape[0]
    grid_h, grid_w = _infer_grid_size(num_patches, image_size, patch_size)

    pca = PCA(n_components=1, svd_solver="randomized")
    first_pc = pca.fit_transform(patch_tokens.astype(np.float32))
    mask = first_pc > threshold

    center_h0 = int(grid_h * border)
    center_h1 = int(grid_h * (1 - border))
    center_w0 = int(grid_w * border)
    center_w1 = int(grid_w * (1 - border))
    center = mask.reshape(grid_h, grid_w)[center_h0:center_h1, center_w0:center_w1]
    if center.sum() <= center.size * min_center_ratio:
        mask = -first_pc > threshold

    mask_2d = mask.reshape(grid_h, grid_w).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_2d = cv2.dilate(mask_2d, kernel).astype(bool)
    mask_2d = cv2.morphologyEx(mask_2d.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return mask_2d.reshape(-1)


def select_foreground_patches(
    patch_tokens,
    image_size,
    patch_size,
    threshold=10.0,
    kernel_size=3,
    border=0.2,
    min_center_ratio=0.35,
):
    mask = compute_foreground_mask(
        patch_tokens=patch_tokens,
        image_size=image_size,
        patch_size=patch_size,
        threshold=threshold,
        kernel_size=kernel_size,
        border=border,
        min_center_ratio=min_center_ratio,
    )
    if mask.sum() == 0:
        mask = np.ones_like(mask, dtype=bool)
        return patch_tokens, mask
    return patch_tokens[mask], mask
