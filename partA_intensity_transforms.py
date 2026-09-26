import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(HERE, "Fig0308(a)(fractured_spine).tif")
OUT_DIR = os.path.join(HERE, "results", "partA")

L = 256

C_LOG_SWEEP_FACTORS = [0.6, 0.8, 1.0, 1.2]

C_POW = 1.0
GAMMA_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
C_POWER_SWEEP_FACTORS = [0.6, 1.0, 1.4]


def load_gray(path):
    img = Image.open(path).convert("L")
    return np.array(img)


def log_transform(img, c=None):
    r = img.astype(np.float64)
    if c is None:
        c = (L - 1) / np.log(1 + r.max())
    s = c * np.log1p(r)
    return to_uint8(s), c


def power_law_transform(img, c=1.0, gamma=1.0):
    r = img.astype(np.float64) / (L - 1)
    s = c * np.power(r, gamma) * (L - 1)
    return to_uint8(s)


def to_uint8(x):
    return np.clip(np.round(x), 0, L - 1).astype(np.uint8)


def save(img, name):
    Image.fromarray(img).save(os.path.join(OUT_DIR, name))


def show_grid(images, titles, fname, ncols=3, suptitle=None):
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, im, t in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=L - 1)
        ax.set_title(t)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = load_gray(IMG_PATH)
    print(f"Loaded {IMG_PATH}: shape={img.shape}, dtype={img.dtype}, "
          f"min={img.min()}, max={img.max()}, mean={img.mean():.1f}")

    save(img, "original.png")

    c_nat = (L - 1) / np.log(1 + float(img.max()))
    log_sweep = [log_transform(img, f * c_nat)[0] for f in C_LOG_SWEEP_FACTORS]
    show_grid([img] + log_sweep,
              ["Original"] + [f"Log, c = {f:.1f} x {c_nat:.1f}" for f in C_LOG_SWEEP_FACTORS],
              "log_sweep.png")
    print(f"Log transform: natural c = {c_nat:.4f}, saved log_sweep.png")

    for f in C_POWER_SWEEP_FACTORS:
        c = f * C_POW
        pow_sweep = [power_law_transform(img, c, g) for g in GAMMA_SWEEP]
        show_grid(pow_sweep, [f"c = {c:g}, gamma = {g}" for g in GAMMA_SWEEP],
                  f"power_sweep_c{c:g}.png",
                  suptitle=f"Power law s = c * r^gamma, c = {c:g}")
        print(f"Power law: saved power_sweep_c{c:g}.png")

    print(f"Results saved to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
