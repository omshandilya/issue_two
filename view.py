"""Visualizer for Squeeze the Scene.
Compares the original SAR image against the reconstructed image side-by-side.

Usage:
    python view.py [path_to_original.npy] [path_to_rebuilt.npy]
Defaults:
    python view.py practice/practice_1.npy squeeze/rebuilt.npy
"""
import os
import sys
import numpy as np

def main():
    orig_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("practice", "practice_1.npy")
    rebuilt_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("squeeze", "rebuilt.npy")

    if not os.path.exists(orig_path):
        print(f"Original file not found: {orig_path}")
        return
    if not os.path.exists(rebuilt_path):
        print(f"Rebuilt file not found: {rebuilt_path}")
        return

    print(f"Loading {orig_path} and {rebuilt_path}...")
    orig = np.load(orig_path)
    rebuilt = np.load(rebuilt_path)

    # Convert to decibels for visual display (raw values are too dark)
    orig_db = 20 * np.log10(np.maximum(orig, 1e-6))
    rebuilt_db = 20 * np.log10(np.maximum(rebuilt, 1e-6))

    vmin = np.percentile(orig_db, 1)
    vmax = np.percentile(orig_db, 99.9)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(orig_db, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Original: {os.path.basename(orig_path)}")
        axes[0].axis("off")

        axes[1].imshow(rebuilt_db, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Rebuilt: {os.path.basename(rebuilt_path)}")
        axes[1].axis("off")

        out_img = "comparison.png"
        plt.tight_layout()
        plt.savefig(out_img, dpi=150)
        print(f"Saved side-by-side comparison to '{out_img}'.")
        print("Displaying interactive window (close window when done)...")
        plt.show()
    except ImportError:
        print("\nNote: matplotlib is not installed. To view interactively:")
        print("   pip install matplotlib")
        print("Then run: python view.py")

if __name__ == "__main__":
    main()

