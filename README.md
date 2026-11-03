# VERITAS-Echo

**Physics-Grounded Multi-Agent Verification for Collapse-Resistant, Self-Improving Cardiac Function AI in Echocardiography**

VERITAS-Echo is a research project focused on verifying cardiac function analysis in echocardiography using physics-grounded checks. The current repository provides a Phase 0 audit utility for EchoNet-Dynamic: it profiles dataset metadata and checks labeled ejection fraction (EF) against EF recomputed from expert volume tracings with Simpson’s method of disks.

> **Current scope:** The code in this repository is a dataset-audit script, not a trained cardiac-function model or a clinical application. Its outputs are for research and data-quality investigation, not diagnosis or treatment.

## What the audit does

`veritas_phase0_audit.py`:

1. Summarizes `FileList.csv`, including split counts, EF distribution, frame dimensions, frame counts, and consistency between EF, EDV, and ESV.
2. Reads `VolumeTracings.csv` and estimates a volume for each traced frame using one long-axis measurement and 20 chord measurements.
3. Selects the maximum- and minimum-volume traced frames as end-diastole (ED) and end-systole (ES), respectively, then computes EF as `(EDV - ESV) / EDV * 100`.
4. Compares this estimate with the dataset EF label and flags videos whose absolute difference exceeds a configurable threshold.
5. Writes a text summary, a per-video CSV, and diagnostic plots. Optionally, it overlays tracing chords on ED/ES video frames.

## Repository layout

```text
.
├── veritas_phase0_audit.py   # Dataset profiling and Simpson’s-disk EF audit
└── results/                  # Example audit outputs and visualizations
```

The committed `results/` files are sample outputs from an audit run; rerunning the script writes fresh files to the selected output directory.

## Requirements

- Python 3
- `pandas`
- `numpy`
- `matplotlib`
- Optional: `opencv-python` when using `--videos` to create tracing overlays
- EchoNet-Dynamic metadata files: `FileList.csv` and `VolumeTracings.csv`
- Optional EchoNet-Dynamic `.avi` videos for frame overlays

The dataset files and videos are not included in this repository. Obtain the dataset through its official distribution and follow its terms of use.

## Run the audit

Install the dependencies:

```bash
pip install pandas numpy matplotlib opencv-python
```

Run from the repository root, supplying the dataset CSV paths:

```bash
python veritas_phase0_audit.py \
  --filelist /path/to/EchoNet-Dynamic/FileList.csv \
  --tracings /path/to/EchoNet-Dynamic/VolumeTracings.csv \
  --out results
```

To also render overlays for the best- and worst-matching videos, provide the directory containing the `.avi` files:

```bash
python veritas_phase0_audit.py \
  --filelist /path/to/EchoNet-Dynamic/FileList.csv \
  --tracings /path/to/EchoNet-Dynamic/VolumeTracings.csv \
  --videos /path/to/EchoNet-Dynamic/Videos \
  --out results
```

The discrepancy threshold defaults to 10 EF percentage points. Change it with `--flag`:

```bash
python veritas_phase0_audit.py \
  --filelist /path/to/EchoNet-Dynamic/FileList.csv \
  --tracings /path/to/EchoNet-Dynamic/VolumeTracings.csv \
  --flag 5
```

Run `python veritas_phase0_audit.py --help` for the available command-line options. The script assumes the tracing CSV contains `FileName`, `Frame`, `X1`, `Y1`, `X2`, and `Y2` columns, and the file list contains fields including `FileName`, `EF`, `EDV`, `ESV`, and `Split`.

## Outputs

The output directory contains:

- `summary.txt` — dataset profile and aggregate EF comparison metrics
- `simpson_ef_vs_label.csv` — per-video ED/ES frames, computed EF, label EF, absolute error, and flag
- `fig_ef_histogram.png` — distribution of label EF by EF band
- `fig_simpson_vs_label.png` — computed EF versus labeled EF, with flagged cases highlighted
- `fig_tracing_<file>.png` — optional ED/ES tracing overlays when `--videos` is used

## Method notes

For each traced frame, the script calculates disk volume as

`V = (π / 4) × Σ(dᵢ²) × (L / N)`

where `L` is the length of the first (long-axis) tracing, `dᵢ` are the remaining chord lengths, and `N` is the number of chords. EF is calculated from the maximum and minimum estimated volumes among the available traced frames for a video. If duplicate tracing rows exist, the script uses the first 21 rows for a frame.

These are implementation choices for this audit and should be considered when interpreting results. In particular, this script does not perform video segmentation or train or evaluate a neural network.

## License and citation

No license or citation instructions are currently provided in this repository. Check the dataset's official documentation for its terms and citation requirements before using or redistributing dataset-derived material. Add project-specific license and citation information here when available.
