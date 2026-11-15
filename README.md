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
├── veritas_phase0_audit.py      # Phase 0: EchoNet-Dynamic profiling + Simpson's-disk EF audit
├── veritas_dataset_audit.py     # Audit for all five datasets (dynamic, pediatric, camus, synthetic)
├── veritas_inner_loop.py        # Inner loop: preprocess -> segment -> volume -> EF -> physics check -> retry
├── veritas_inner_loop_visual.py # Animated GIF of the inner loop on one video, for demos
├── veritas_outer_loop.py        # Outer loop data side: capped mixer, test-leak guard, Tail Guard, tail replay
├── veritas_demo.py              # Runs everything above in one command, for reviews
├── results/                     # Example Phase 0 audit outputs and visualizations
└── docs/
    ├── requirements/            # SRS for the inner loop and the outer loop
    ├── architecture/            # SAD for the inner loop and the outer loop
    ├── test-plans/              # Test plans for the inner loop and the outer loop
    └── report/                  # ESA report, Chapter 7: Implementation
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


## Quick demo

```bash
pip install pandas numpy matplotlib opencv-python
python veritas_demo.py                                                  # self-tests only, no data needed
python veritas_demo.py --filelist FileList.csv --tracings VolumeTracings.csv   # adds the real-data audit
```

## Dataset audit (`veritas_dataset_audit.py`)

```bash
python veritas_dataset_audit.py --dataset dynamic   --filelist FileList.csv --tracings VolumeTracings.csv
python veritas_dataset_audit.py --dataset pediatric --filelist FileList.csv --tracings VolumeTracings.csv --duplicate-tail 2
python veritas_dataset_audit.py --dataset camus     --camus-root /path/to/CAMUS/training     # needs SimpleITK
python veritas_dataset_audit.py --dataset synthetic --real-video real.avi --synthetic-video synth.avi \
                                --labels synth_labels.csv --filelist FileList.csv
```

- Every mode prints an attribute-relevance table: which columns are used and why.
- `pediatric` reports EF error by age band, sex and BMI; `camus` by sex, age band, image quality and the EF <= 40% tail.
- `synthetic` compares real and synthetic data with a KS distance on EF and an intensity-histogram overlap.
- Verified on real data: `dynamic`. Tested on synthetic stand-in data only: `pediatric`, `camus`, `synthetic`.

## Inner loop (`veritas_inner_loop.py`)

```bash
python veritas_inner_loop.py --selftest
python veritas_inner_loop.py --calibrate --tracings VolumeTracings.csv --filelist FileList.csv
python veritas_inner_loop.py --video clip.avi --sex F --age 54
python veritas_inner_loop_visual.py --video clip.avi --sex F --age 54   # writes clip_inner_loop.gif (needs matplotlib, pillow)
```

- Heart rate is read from the volume curve and checked against age-dependent bounds that accept an athlete's ~40 bpm.
- A verified EF is interpreted with the ASE/EACVI 2015 sex-specific ranges (men 52-72%, women 54-74%); adult cut-offs are not applied under 18.
- `--calibrate` gives a lower bound for kappa_max from the expert tracings: on EchoNet-Dynamic, 0.074 of EDV per frame (ED to ES takes a median of 16 frames).
- The `classical` segmentation backend is a threshold placeholder, **not** SAM-2. The `sam2` backend is a marked integration point.

## Outer loop (`veritas_outer_loop.py`)

```bash
python veritas_outer_loop.py --selftest
python veritas_outer_loop.py --filelist FileList.csv --synthetic-labels synth.csv --ratio 0.05 --replay-every 20
```

- Synthetic share is capped at 75%; a fully synthetic generation is refused.
- Videos in the TEST split can never enter a training index.
- The Tail Guard restores a minimum EF <= 40% share after filtering, and tail videos are replayed every K items.
- Not built yet: learned verifiers V1-V3, LLM auditor V5, ensemble V6, retraining, Collapse Observatory.
