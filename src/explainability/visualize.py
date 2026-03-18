import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.explainability.gradcam import compute_gradcam


def overlay_gradcam(model, img_array, layer_name: str,
                    img_display=None, title: str = "", ax=None):
    """
    Compute GradCAM and overlay the heatmap on the original image.

    Args:
        model:       Trained Keras model.
        img_array:   np.ndarray (1, H, W, 1), values in [0, 1].
        layer_name:  Target conv layer name.
        img_display: Optional (H, W) grayscale array for display.
                     Defaults to img_array.squeeze().
        title:       Plot title.
        ax:          Matplotlib axis. If None, a new figure is created.
    """
    heatmap = compute_gradcam(model, img_array, layer_name)

    H, W = img_array.shape[1], img_array.shape[2]
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # Convert heatmap to RGB jet colormap
    heatmap_rgb = plt.cm.jet(heatmap_resized)[:, :, :3]      # (H, W, 3)

    # Base image as RGB
    base = img_display if img_display is not None else img_array.squeeze()
    base_rgb = np.stack([base] * 3, axis=-1)                  # (H, W, 3)

    # Blend
    overlay = 0.5 * base_rgb + 0.4 * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)

    show = ax is None
    if show:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(base, cmap="gray");        ax[0].set_title("Original");  ax[0].axis("off")
    ax[1].imshow(heatmap_resized, cmap="jet"); ax[1].set_title("GradCAM"); ax[1].axis("off")
    ax[2].imshow(overlay);                  ax[2].set_title("Overlay");   ax[2].axis("off")

    if title:
        plt.suptitle(title, fontsize=11)

    if show:
        plt.tight_layout()
        plt.show()


def visualize_gradcam_batch(model, test_gen, layer_name: str,
                             n: int = 6, save_path: str = None):
    """
    Show GradCAM overlays for n random images from test_gen.

    Args:
        model:      Trained Keras model.
        test_gen:   Keras ImageDataGenerator flow.
        layer_name: Target conv layer name.
        n:          Number of images to display.
        save_path:  If provided, saves the figure to this path.
    """
    classes = list(test_gen.class_indices.keys())
    test_gen.reset()
    images, labels = next(test_gen)
    images = images[:n]
    labels = labels[:n]

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    for i, (img, lbl) in enumerate(zip(images, labels)):
        img_array = np.expand_dims(img, axis=0)               # (1, H, W, 1)
        prob  = model.predict(img_array, verbose=0)[0][0]
        pred  = classes[int(prob >= 0.5)]
        truth = classes[int(lbl)]
        title = f"True: {truth} | Pred: {pred} ({prob:.2%})"
        overlay_gradcam(model, img_array, layer_name,
                        img_display=img.squeeze(),
                        title=title, ax=axes[i])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()
