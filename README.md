# ECE1512 — Assignment 1

Intensity transformations and histogram equalization on a grayscale radiograph, implemented
from first principles with NumPy (no OpenCV, no `skimage.exposure`).

**Input image:** `Fig0308(a)(fractured_spine).tif` — 976 × 746 grayscale, full 0–255 range but
a mean intensity of only 32.6, so it is heavily skewed toward black. That skew is what makes
both parts of the assignment visible: the image has detail hiding in the dark end.

## Layout

| Path | What it is |
| --- | --- |
| `partA_intensity_transforms.py` | Part A — log and power-law (gamma) transforms |
| `partB_histogram_equalization.py` | Part B — histogram equalization |
| `Fig0308(a)(fractured_spine).tif` | Input image, read by both scripts |
| `results/partA/`, `results/partB/` | Generated figures (committed, so no need to rerun) |
| `requirements.txt` | numpy, pillow, matplotlib |

Both scripts resolve paths relative to their own file, so they can be run from any working
directory.

## Running

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python partA_intensity_transforms.py
python partB_histogram_equalization.py
```

Each script prints image statistics to the console and writes its figures into
`results/`, overwriting whatever was there.

If `Activate.ps1` is blocked by execution policy, either call the interpreter directly
(`.\.venv\Scripts\python.exe partA_intensity_transforms.py`) or allow local scripts for the
session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## Part A — intensity transformations

**Log transform**, `s = c · log(1 + r)`, computed in float64 via `np.log1p` and then clipped
and rounded back to uint8.

The reference scaling is `c_nat = (L−1) / log(1 + max(r))`, the value that maps the brightest
input pixel exactly to 255. `log_sweep.png` shows the original alongside four multiples of
that value — `c = {0.6, 0.8, 1.0, 1.2} × c_nat` — so the effect of under- and over-scaling is
visible side by side. Above `c_nat` the bright end saturates; below it the output never
reaches full white.

**Power-law (gamma) transform**, `s = c · r^γ`, with `r` normalized to [0, 1] before
exponentiation and rescaled to [0, 255] after. Only `γ < 1` is swept
(`γ = 0.2 … 0.8`), since this image needs *brightening*; `γ > 1` would push an already-dark
image darker.

Three figures cover the gain term: `power_sweep_c0.6.png`, `power_sweep_c1.png`,
`power_sweep_c1.4.png`. Each holds `c` fixed and varies `γ` across six panels, which
separates the two knobs — `γ` reshapes contrast, `c` only scales the result (and at `c = 1.4`
clips the highlights).

Clipping is deliberate and centralized in `to_uint8()`: `clip → round → uint8`. Rounding
before clipping, or letting float values wrap into uint8, is the usual source of
salt-and-pepper artifacts in the bright regions here.

## Part B — histogram equalization

Implements the textbook transformation directly:

```
T(r_k) = round( (L−1) · Σ_{j=0..k} p_r(r_j) ),    p_r(r_j) = n_j / MN
```

Three steps, one function each:

1. `compute_histogram()` — counts pixels per intensity with an explicit `i, j` loop over the
   array. `np.bincount` would be far faster, but the loop is the point: it shows the
   `n_j / MN` accounting the assignment asks for.
2. `equalization_transform()` — accumulates `p_r` into a 256-entry uint8 lookup table.
3. `apply_transform()` — `T[img]`, NumPy fancy indexing, which maps every pixel in one shot.

The histogram of the *output* is recomputed with the same function rather than derived
analytically, so the before/after comparison is measured, not assumed.

`summary.png` is the figure to look at: original and equalized images, the transformation
curve `T(r)` plotted against the identity line, and normalized histograms of both images.
`original.png` and `equalized.png` are the two images on their own.

The expected result is the standard one — equalization spreads the crowded dark end across
the full range, raising mean intensity substantially and pulling out spine detail, while
leaving characteristic gaps in the output histogram because a discrete `T` maps several input
levels onto the same output level and can never truly flatten it.

## Notes

- `.venv/` and `__pycache__/` are not tracked.
- The figures in `results/` are committed so the output can be reviewed without running
  anything. They are regenerated on every run, so expect them to show up as changes.
