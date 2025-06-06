import numpy as np
import matplotlib.pyplot as plt
import shap
from matplotlib.colors import Normalize
import argparse
import os

def load_shap_data(path):
    data = np.load(path, allow_pickle=True)
    shap_values = data["shap_values"]  # (1, 3, H, W)
    image = data["input_image"]        # (H, W, 3)
    class_name = str(data["class_name"])
    image_path = str(data["image_path"])
    return shap_values, image, class_name, image_path

def visualize_shap_overlay(shap_values, image, class_name):
    # Extract and average across RGB channels
    shap_array = shap_values[0]  # (3, H, W)
    shap_map = shap_array.mean(0)  # preserve sign! shape: (H, W)

    # Normalize: use diverging colormap centered at 0
    vmin = np.percentile(shap_map, 1)
    vmax = np.percentile(shap_map, 99)
    vmax = max(abs(vmin), abs(vmax))  # symmetric range around 0
    vmin = -vmax

    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.imshow(shap_map, cmap="seismic", norm=norm, alpha=0.5)
    plt.title(f"SHAP Overlay for '{class_name}'", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_shap_native(shap_values, image):
    # Rescale image if needed
    img = image.astype(np.float32) / 255.0
    shap.image_plot(shap_values, np.array([img]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to .npz SHAP file")
    parser.add_argument("--mode", type=str, default="none", choices=["overlay", "shap", "none"], help="Visualization mode")
    args = parser.parse_args()

    shap_values, image, class_name, image_path = load_shap_data(args.file)

    print(f"✅ Loaded SHAP file for class '{class_name}' from image {image_path}")
    print(f"  SHAP values shape: {shap_values.shape}")
    print(f"  Input image shape: {image.shape}")

    vals = shap_values[0, 0]  # shape: (3, H, W)
    print("SHAP min/max per channel:", [(v.min(), v.max()) for v in vals])

    # if args.mode == "overlay":
    #     visualize_shap_overlay(shap_values, image, class_name)
    # elif args.mode == "shap":
    #     visualize_shap_native(shap_values, image)

if __name__ == "__main__":
    main()
