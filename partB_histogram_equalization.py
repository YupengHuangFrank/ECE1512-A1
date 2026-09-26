import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(HERE, "Fig0308(a)(fractured_spine).tif")
OUT_DIR = os.path.join(HERE, "results", "partB")

L = 256 


def load_gray(path):
    """Read an image file as a 2-D uint8 grayscale array."""
    return np.array(Image.open(path).convert("L"))


def compute_histogram(img):
    """Count how many pixels have each intensity 0..L-1 (explicit pixel loop)."""
    hist = np.zeros(L, dtype=np.int64)
    M, N = img.shape
    for i in range(M):
        for j in range(N):
            hist[img[i, j]] += 1
    return hist


def equalization_transform(hist):
    """T(r_k) = round((L-1) * sum_{j=0..k} p_r(r_j))  -> lookup table of length L."""
    total = hist.sum()
    T = np.zeros(L, dtype=np.uint8)
    cumulative = 0.0
    for k in range(L):
        cumulative += hist[k] / total          # running sum of p_r(r_j)
        T[k] = int(round((L - 1) * cumulative))
    return T


def apply_transform(img, T):
    """Map every pixel r to T[r]."""
    return T[img]


def save_image(img, name):
    Image.fromarray(img).save(os.path.join(OUT_DIR, name))


def plot_summary(img, eq, hist, hist_eq, T, fname):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(img, cmap="gray", vmin=0, vmax=L - 1)
    axes[0, 0].set_title("Original")
    axes[0, 1].plot(np.arange(L), T)
    axes[0, 1].plot([0, L - 1], [0, L - 1], "k--", linewidth=0.8)
    axes[0, 1].set_title("Transformation T(r)")
    axes[0, 1].set_xlabel("Input intensity r_k")
    axes[0, 1].set_ylabel("Output intensity s_k")
    axes[0, 1].set_xlim(0, L - 1); axes[0, 1].set_ylim(0, L - 1)
    axes[0, 2].imshow(eq, cmap="gray", vmin=0, vmax=L - 1)
    axes[0, 2].set_title("Equalized")
    axes[1, 0].bar(np.arange(L), hist / hist.sum(), width=1.0, color="0.25")
    axes[1, 0].set_title("Histogram of original")
    axes[1, 2].bar(np.arange(L), hist_eq / hist_eq.sum(), width=1.0, color="0.25")
    axes[1, 2].set_title("Histogram of equalized")
    for ax in (axes[1, 0], axes[1, 2]):
        ax.set_xlim(-1, L)
        ax.set_xlabel("Intensity r_k")
        ax.set_ylabel("p(r_k)")
    for ax in (axes[0, 0], axes[0, 2]):
        ax.axis("off")
    axes[1, 1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = load_gray(IMG_PATH)
    print(f"Loaded image: shape={img.shape}, min={img.min()}, max={img.max()}")

    hist = compute_histogram(img)
    T = equalization_transform(hist)
    eq = apply_transform(img, T)
    hist_eq = compute_histogram(eq)

    save_image(img, "original.png")
    save_image(eq, "equalized.png")
    plot_summary(img, eq, hist, hist_eq, T, "summary.png")

    print("T(r) for r = 0..5:", T[:6].tolist())
    print(f"Mean intensity: original {img.mean():.1f}, equalized {eq.mean():.1f}")
    print(f"Results saved to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
