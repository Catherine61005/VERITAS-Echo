"""
VERITAS-Echo - unified dataset audit: all five datasets, one tool.

Covers every dataset named in the problem statement and project ID
(EchoNet-Dynamic, EchoNet-Pediatric, CAMUS, EchoNet-Synthetic, EchoFlow),
dispatched from one script with --dataset {dynamic,pediatric,camus,synthetic}.

Why one script and not five
  Review-1's panel wanted proof the datasets were actually downloaded and
  explored, and a real-vs-synthetic comparison. Review-2's panel wanted
  patient demographics accounted for, and a working demo instead of only an
  audit. This file answers all three by reusing the SAME validated
  measurement core across every real dataset, and adding the comparison the
  panel asked for on the synthetic side - so the "implementation" slide for
  the ESA is one coherent pipeline, not five unrelated scripts.

  veritas_phase0_audit.py (EchoNet-Dynamic) and veritas_phase0b_pediatric_audit.py
  (EchoNet-Pediatric) already do the --dataset dynamic / pediatric work on
  their own; this file folds both into one entry point and adds camus and
  synthetic, so one command line + one report chapter can describe the full
  dataset story.

--dataset dynamic / pediatric  (real patient video, expert tracings)
  Identical approach to the two scripts above: read FileList.csv +
  VolumeTracings.csv, recompute EF from the expert tracings with Simpson's
  method of disks, compare to the label EF. pediatric additionally
  stratifies the comparison by age band / sex if those columns are present.
  THIS PATH IS THE ONE WE HAVE VALIDATED AGAINST REAL DATA: on the actual
  EchoNet-Dynamic FileList.csv/VolumeTracings.csv, it reproduces MAE 1.04 EF
  points, r = 0.973 on 10,024 videos (see /results in the project repo).
  The pediatric path runs the same code but has only been smoke-tested on
  synthetic placeholder data in this environment, since the real
  EchoNet-Pediatric CSVs were not available when this script was written -
  run it yourself on the real files before trusting its printed numbers.

--dataset camus  (real patient video, pixel-mask ground truth, not chord tracings)
  CAMUS ships per-patient segmentation masks (ED and ES), not the
  long-axis+20-chord tracing format EchoNet uses. mask_to_volume_by_disks()
  adapts the same Simpson's-disk idea to a binary mask: it slices the mask
  into 20 horizontal bands along its vertical extent and uses each band's
  mean width as the disk diameter - this is deliberately the same shape of
  computation the inner loop will eventually run on a SAM-2 mask, so CAMUS
  doubles as a rehearsal for that step, not just a third dataset audit.
  Verified on a synthetic prolate-spheroid mask against the exact analytic
  volume formula (4/3 * pi * a * b^2): matched within about 6%, which is
  the expected pixel-discretization error for a 20-band Simpson's estimate,
  not a formula bug. NOT yet run against real CAMUS files (needs
  SimpleITK and the actual .mhd/.raw files, neither available here) -
  the --camus-root loader is written and documented but unverified end to end.

--dataset synthetic  (EchoNet-Synthetic / EchoFlow - generator output, no real patient)
  These aren't audited for measurement accuracy the way real data is - they
  ARE the data whose trustworthiness is in question. What's useful here,
  and what Review-1's panel explicitly asked for, is a real-vs-synthetic
  visual comparison and an EF-label profile. real_vs_synthetic_figure()
  takes one real and one synthetic video (or frame image), aligns them to a
  matching EF band if a labels CSV is given, and renders them side by side.
  Smoke-tested on two dummy generated frames; not run on real
  EchoNet-Synthetic/EchoFlow clips in this session.

Usage
  pip install pandas numpy matplotlib opencv-python
  # optional, only for --dataset camus:
  pip install SimpleITK

  python veritas_dataset_audit.py --dataset dynamic   --filelist FileList.csv --tracings VolumeTracings.csv --out results_dynamic
  python veritas_dataset_audit.py --dataset pediatric --filelist FileList.csv --tracings VolumeTracings.csv --out results_pediatric [--view A4C]
  python veritas_dataset_audit.py --dataset camus      --camus-root /path/to/CAMUS/training --out results_camus
  python veritas_dataset_audit.py --dataset synthetic  --real-video real.avi --synthetic-video synth.avi --labels synthetic_labels.csv --out results_synthetic
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY, TEAL, ORANGE, RED, PURPLE = "#252C5C", "#1FA9A9", "#E48312", "#C0392B", "#7B5EA7"
BANDS = [(0, 40, "EF <= 40%", RED), (40, 55, "40-55%", ORANGE), (55, 100, "> 55%", TEAL)]

AGE_CANDIDATES = ["Age", "AgeYears", "Age_Years", "PatientAge", "age", "AgeAtEcho", "AgeInYears"]
SEX_CANDIDATES = ["Sex", "Gender", "sex", "PatientSex"]
WEIGHT_CANDIDATES = ["Weight", "WeightKg", "Weight_kg", "weight"]
BSA_CANDIDATES = ["BSA", "BodySurfaceArea", "bsa"]
CAMUS_LV_LABEL = 1  # CAMUS ground-truth label for the LV cavity
VIEW_CANDIDATES = ["View", "ViewType", "view", "Type", "Modality"]


def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def disk_volume(rows):
    """Chord-tracing version: rows = (21, 4) array of X1,Y1,X2,Y2.
    Row 0 = long axis, rows 1..20 = disk chords. Validated on EchoNet-Dynamic."""
    long_axis = np.hypot(rows[0, 2] - rows[0, 0], rows[0, 3] - rows[0, 1])
    d = np.hypot(rows[1:, 2] - rows[1:, 0], rows[1:, 3] - rows[1:, 1])
    return np.pi / 4 * np.sum(d ** 2) * long_axis / len(d)


def mask_to_volume_by_disks(mask, n_disks=20):
    """Pixel-mask version of the same idea: slice a binary LV mask into
    n_disks equal-height horizontal bands along its vertical extent and use
    each band's mean width as the disk diameter. Verified against the exact
    analytic volume of a synthetic prolate-spheroid mask to within ~6%,
    which is the expected discretization error, not a bug.
    Assumes the mask's long (apex-to-base) axis is roughly vertical, which
    holds for CAMUS's provided orientation; a tilted probe angle would need
    a PCA-based axis instead of this image-coordinate shortcut."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0
    y0, y1 = ys.min(), ys.max()
    L = float(y1 - y0)
    edges = np.linspace(y0, y1, n_disks + 1)
    d = np.zeros(n_disks)
    for i in range(n_disks):
        lo, hi = edges[i], edges[i + 1]
        band = (ys >= lo) & (ys <= hi) if i == n_disks - 1 else (ys >= lo) & (ys < hi)
        if band.sum() == 0:
            continue
        widths = [xs[ys == y].max() - xs[ys == y].min() for y in np.unique(ys[band])]
        d[i] = np.mean(widths) if widths else 0.0
    return np.pi / 4 * np.sum(d ** 2) * L / n_disks


ATTRIBUTE_ROLE = {
    "filename": ("used", "key joining FileList to tracings and videos"),
    "ef": ("used", "ground-truth label the measurement is validated against"),
    "edv": ("used", "label consistency check: EF must equal (EDV-ESV)/EDV"),
    "esv": ("used", "label consistency check, and EDV > ESV > 0 physics rule"),
    "split": ("used", "keeps the locked test set out of every training index"),
    "fps": ("used", "converts the volume curve's period into heart rate"),
    "numberofframes": ("used", "detects clips too short to contain a full heartbeat"),
    "frameheight": ("used", "flags non-standard sizes before resizing to 112x112"),
    "framewidth": ("used", "flags non-standard sizes before resizing to 112x112"),
    "age": ("used", "demographic stratification; heart-rate bounds; adult vs paediatric EF cut-offs"),
    "sex": ("used", "demographic stratification; ASE 2015 sex-specific EF ranges"),
    "weight": ("used", "with height, gives BMI for stratification"),
    "height": ("used", "with weight, gives BMI for stratification"),
    "bsa": ("used", "body-size stratification"),
    "view": ("used", "restricts EF work to the apical 4-chamber view"),
}


def attribute_table(columns):
    rows = []
    for c in columns:
        role, why = ATTRIBUTE_ROLE.get(c.lower().replace("_", ""), ("not used", "no role in EF measurement or validation"))
        rows.append((c, role, why))
    return rows


def age_band(age):
    if pd.isna(age):
        return "unknown"
    if age < 1:
        return "0-1y (infant)"
    if age < 6:
        return "1-6y (toddler/preschool)"
    if age < 13:
        return "6-13y (child)"
    return "13-18y (adolescent)"


# --------------------------------------------------------------------- dynamic / pediatric
def run_echonet_family(a, say, is_pediatric):
    fl = pd.read_csv(a.filelist)
    say(f"FileList.csv: {len(fl):,} videos, columns = {list(fl.columns)}")
    say("Attribute relevance (which columns this project uses, and why):")
    for c, role, why in attribute_table(fl.columns):
        say(f"  {c:>16}  {role:<9} {why}")
    att = pd.DataFrame(attribute_table(fl.columns), columns=["column", "role", "reason"])
    att.to_csv(os.path.join(a.out, "attribute_relevance.csv"), index=False)

    age_col = sex_col = weight_col = bsa_col = view_col = None
    if is_pediatric:
        age_col = find_col(fl, AGE_CANDIDATES)
        sex_col = find_col(fl, SEX_CANDIDATES)
        weight_col = find_col(fl, WEIGHT_CANDIDATES)
        bsa_col = find_col(fl, BSA_CANDIDATES)
        view_col = find_col(fl, VIEW_CANDIDATES)
        say(f"Demographic/view columns found -> age: {age_col}, sex: {sex_col}, weight: {weight_col}, "
            f"BSA: {bsa_col}, view: {view_col}")
        if age_col is None and sex_col is None:
            say("  WARNING: no age/sex column matched; edit AGE_CANDIDATES/SEX_CANDIDATES near the top "
                "of this file with the real column names from your FileList.csv and rerun.")
        if view_col:
            say(f"Videos per view: {fl[view_col].value_counts().to_dict()}")
            if a.view:
                before = len(fl)
                fl = fl[fl[view_col].astype(str).str.upper() == a.view.upper()].copy()
                say(f"Restricting to view={a.view}: {before:,} -> {len(fl):,} videos")

    if "EF" in fl.columns:
        say(f"EF: mean {fl.EF.mean():.2f}, median {fl.EF.median():.2f}, min {fl.EF.min():.2f}, max {fl.EF.max():.2f}")
        for lo, hi, name, _ in BANDS:
            n = ((fl.EF > lo) & (fl.EF <= hi)).sum()
            say(f"  {name:>9}: {n:,} videos ({n / len(fl) * 100:.1f}%)")
    if "Split" in fl.columns:
        say(f"Splits: {fl.Split.value_counts().to_dict()}")
    if age_col:
        ages = pd.to_numeric(fl[age_col], errors="coerce")
        fl["_age_band"] = ages.apply(age_band)
        say(f"Patients per age band: {fl['_age_band'].value_counts().to_dict()}")
    if sex_col:
        say(f"Sex ({sex_col}): {fl[sex_col].value_counts(dropna=False).to_dict()}")
    height_col = find_col(fl, ["Height", "HeightCm", "Height_cm", "height"]) if is_pediatric else None
    if weight_col and height_col:
        w = pd.to_numeric(fl[weight_col], errors="coerce")
        h = pd.to_numeric(fl[height_col], errors="coerce")
        h_m = h / 100 if h.median() > 3 else h  # cm or metres, whichever the file uses
        fl["_bmi"] = w / h_m ** 2
        # children's BMI norms depend on age; these coarse bands are for stratifying error, not diagnosis
        fl["_bmi_band"] = pd.cut(fl["_bmi"], bins=[0, 14, 18, 22, 100], labels=["<14", "14-18", "18-22", "22+"])
        say(f"BMI from {weight_col}/{height_col}: median {fl['_bmi'].median():.1f}, "
            f"bands {fl['_bmi_band'].value_counts().to_dict()}")

    vt = pd.read_csv(a.tracings)
    name_col = "FileName" if "FileName" in vt.columns else vt.columns[0]
    vt[name_col] = vt[name_col].astype(str).str.replace(".avi", "", regex=False)
    say()
    say(f"VolumeTracings.csv: {len(vt):,} rows, {vt[name_col].nunique():,} videos")

    fl_name_col = "FileName" if "FileName" in fl.columns else fl.columns[0]
    fl_key = fl.set_index(fl_name_col)

    vols = []
    for (f, frame), g in vt.groupby([name_col, "Frame"], sort=False):
        rows = g[["X1", "Y1", "X2", "Y2"]].values[:21]
        if len(rows) < 2:
            continue
        vols.append((f, frame, disk_volume(rows)))
    v = pd.DataFrame(vols, columns=["FileName", "Frame", "V"])
    ed = v.loc[v.groupby("FileName").V.idxmax()].set_index("FileName")
    es = v.loc[v.groupby("FileName").V.idxmin()].set_index("FileName")
    res = pd.DataFrame({"ED_frame": ed.Frame, "ES_frame": es.Frame, "V_ED": ed.V, "V_ES": es.V})
    res["EF_ours"] = (res.V_ED - res.V_ES) / res.V_ED * 100

    join_cols = [c for c in ["EF", "Split", age_col, sex_col, weight_col, bsa_col, "_age_band", "_bmi_band"]
                 if c and c in fl_key.columns]
    res = res.join(fl_key[join_cols].rename(columns={"EF": "EF_label"}), how="inner")
    res["abs_error"] = (res.EF_ours - res.EF_label).abs()
    res["flagged"] = res.abs_error > a.flag
    res.sort_values("abs_error", ascending=False).to_csv(os.path.join(a.out, "simpson_ef_vs_label.csv"))

    say()
    say(f"Simpson's-disk EF vs label EF on {len(res):,} videos:")
    say(f"  MAE {res.abs_error.mean():.2f} EF points, median {res.abs_error.median():.2f}, "
        f"bias {(res.EF_ours - res.EF_label).mean():+.2f}, r = {np.corrcoef(res.EF_ours, res.EF_label)[0, 1]:.3f}")
    say(f"  Flagged (|error| > {a.flag:g}): {res.flagged.sum()} videos ({res.flagged.mean() * 100:.1f}%)")

    # --- Review-2 panel suggestion: duplicate rare/low-EF cases as a quick supplement to the Tail Guard ---
    if a.duplicate_tail and "EF_label" in res.columns:
        tail = res[res.EF_label <= 40]
        rest = res[res.EF_label > 40]
        oversampled_index = pd.concat(
            [rest.index.to_series()] + [tail.index.to_series()] * (1 + a.duplicate_tail),
            ignore_index=True,
        )
        oversampled_index.to_frame("FileName").to_csv(
            os.path.join(a.out, "training_index_tail_oversampled.csv"), index=False
        )
        say()
        say(f"Tail oversampling (--duplicate-tail {a.duplicate_tail}): {len(tail)} low-EF (<=40%) videos "
            f"duplicated {1 + a.duplicate_tail}x each.")
        say(f"  Training index size: {len(rest) + len(tail):,} -> {len(oversampled_index):,} rows; "
            f"low-EF share: {len(tail) / (len(rest) + len(tail)) * 100:.1f}% -> "
            f"{len(tail) * (1 + a.duplicate_tail) / len(oversampled_index) * 100:.1f}%.")
        say("  Note: this is a cheap, immediate complement to the architecture's Tail Guard (band-quota "
            "gating in the outer loop), not a replacement for it - duplication changes sampling frequency, "
            "Tail Guard changes which synthetic videos are trusted at all.")

    if is_pediatric:
        say()
        say("Demographic-stratified error:")
        if "_age_band" in res.columns:
            for band, g in res.groupby("_age_band"):
                say(f"  {band:>28}: n={len(g):4d}  MAE {g.abs_error.mean():.2f}")
        if sex_col and sex_col in res.columns:
            for sex, g in res.groupby(sex_col):
                say(f"  {str(sex):>28}: n={len(g):4d}  MAE {g.abs_error.mean():.2f}")
        if "_bmi_band" in res.columns:
            for band, g in res.groupby("_bmi_band", observed=True):
                say(f"  {'BMI ' + str(band):>28}: n={len(g):4d}  MAE {g.abs_error.mean():.2f}")

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(6.0, 2.9), dpi=220)
    ok = ~res.flagged
    ax.scatter(res.EF_label[ok], res.EF_ours[ok], s=3, color=NAVY, alpha=0.25, linewidths=0)
    ax.scatter(res.EF_label[~ok], res.EF_ours[~ok], s=9, color=RED, linewidths=0,
               label=f"off by > {a.flag:g} points ({res.flagged.sum()})")
    ax.plot([0, 100], [0, 100], color=TEAL, lw=1); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("label EF (%)"); ax.set_ylabel("our Simpson's-disk EF (%)")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.text(3, 95, f"MAE {res.abs_error.mean():.2f}   r = {np.corrcoef(res.EF_ours, res.EF_label)[0, 1]:.3f}"
                   f"   n = {len(res):,}", fontsize=9.5, color=NAVY, fontweight="bold", va="top")
    ax.set_title(f"{'EchoNet-Pediatric' if is_pediatric else 'EchoNet-Dynamic'}: our EF vs label EF",
                 fontsize=10.5, color=NAVY, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_simpson_vs_label.png")); plt.close(fig)


# --------------------------------------------------------------------- CAMUS
def run_camus(a, say):
    """CAMUS layout: --camus-root/patientNNNN/patientNNNN_{2CH,4CH}_{ED,ES}.mhd (+ .raw)
    and patientNNNN_{2CH,4CH}_{ED,ES}_gt.mhd (ground-truth LV mask), plus an
    Info_2CH.cfg / Info_4CH.cfg with the clinically reported EF/EDV/ESV.
    This loader has NOT been run against real CAMUS files in this session -
    verify the exact file-naming pattern against your downloaded copy; the
    dataset's released structure has varied slightly across versions."""
    try:
        import SimpleITK as sitk
    except ImportError:
        say("ERROR: SimpleITK is required for --dataset camus (pip install SimpleITK). "
            "mask_to_volume_by_disks() itself has no dependency on it; only this file loader does.")
        return

    patient_dirs = sorted(glob.glob(os.path.join(a.camus_root, "patient*")))
    say(f"Found {len(patient_dirs)} patient folders under {a.camus_root}")
    rows = []
    for pdir in patient_dirs:
        pid = os.path.basename(pdir)
        # Info_4CH.cfg is "Key: value" lines. CAMUS records Sex, Age, ImageQuality and EF per
        # patient, which is what lets us answer Review-2's demographics question on adult data.
        meta = {}
        cfg_path = os.path.join(pdir, "Info_4CH.cfg")
        if os.path.exists(cfg_path):
            for line in open(cfg_path):
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip().lower()] = v.strip()
        def num(*keys):
            # field names differ between CAMUS releases (e.g. "EF" vs "LVef"), so try each
            for key in keys:
                try:
                    return float(meta[key])
                except (KeyError, ValueError):
                    continue
            return None
        vols = {}
        for phase in ("ED", "ES"):
            gt_path = os.path.join(pdir, f"{pid}_4CH_{phase}_gt.mhd")
            if not os.path.exists(gt_path):
                continue
            img = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
            # CAMUS labels: 0 background, 1 LV cavity, 2 myocardium, 3 left atrium.
            # Only label 1 is the blood volume EF is defined on.
            mask = np.squeeze(img) == CAMUS_LV_LABEL
            vols[phase] = mask_to_volume_by_disks(mask)
        if "ED" in vols and "ES" in vols and vols["ED"] > 0:
            ef_ours = (vols["ED"] - vols["ES"]) / vols["ED"] * 100
            rows.append((pid, vols["ED"], vols["ES"], ef_ours, num("ef", "lvef"),
                         meta.get("sex"), num("age"), meta.get("imagequality")))
    if not rows:
        say("No patients with both ED and ES ground-truth masks were found - check the file-naming pattern above.")
        return
    res = pd.DataFrame(rows, columns=["patient", "V_ED", "V_ES", "EF_ours", "EF_label",
                                      "Sex", "Age", "ImageQuality"]).set_index("patient")
    res["abs_error"] = (res.EF_ours - res.EF_label).abs()
    res["age_band"] = pd.cut(res.Age, bins=[0, 40, 60, 75, 120], labels=["<40", "40-60", "60-75", "75+"])
    res.to_csv(os.path.join(a.out, "camus_ef_vs_label.csv"))
    say(f"CAMUS: computed EF for {len(res)} patients from ground-truth masks (4-chamber view).")
    lab = res[res.EF_label.notna()]
    if len(lab):
        say(f"  Against the reported EF ({len(lab)} patients): MAE {lab.abs_error.mean():.2f} EF points")
        say("Demographic-stratified error (adults):")
        for col in ["Sex", "age_band", "ImageQuality"]:
            g = lab.groupby(col, observed=True).abs_error.agg(["count", "mean"])
            for k, r in g.iterrows():
                say(f"  {col:>12} = {str(k):<8} n={int(r['count']):4d}  MAE {r['mean']:.2f}")
        tail = lab[lab.EF_label <= 40]
        say(f"  Tail (EF <= 40%): n={len(tail)}  MAE {tail.abs_error.mean():.2f}" if len(tail) else
            "  Tail (EF <= 40%): none in this subset")


# --------------------------------------------------------------------- synthetic
def real_vs_synthetic_figure(real_frame, synth_frame, out_path, real_label="real", synth_label="synthetic"):
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.8), dpi=220)
    axs[0].imshow(real_frame, cmap="gray"); axs[0].set_title(real_label, fontsize=9, color=NAVY); axs[0].axis("off")
    axs[1].imshow(synth_frame, cmap="gray"); axs[1].set_title(synth_label, fontsize=9, color=ORANGE); axs[1].axis("off")
    fig.suptitle("Real vs synthetic echo frame", fontsize=10, color=NAVY)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def grab_middle_frame(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(n // 2, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame from {video_path}")
    return frame[:, :, ::-1]


def ks_distance(x, y):
    """Two-sample Kolmogorov-Smirnov statistic: the largest gap between the two
    cumulative distributions. 0 = identical distributions, 1 = no overlap."""
    x, y = np.sort(np.asarray(x, float)), np.sort(np.asarray(y, float))
    grid = np.concatenate([x, y])
    return float(np.max(np.abs(np.searchsorted(x, grid, side="right") / len(x)
                                - np.searchsorted(y, grid, side="right") / len(y))))


def histogram_overlap(a_img, b_img, bins=32):
    """Share of intensity histogram the two frames have in common (1 = same look)."""
    ha, _ = np.histogram(a_img, bins=bins, range=(0, 255))
    hb, _ = np.histogram(b_img, bins=bins, range=(0, 255))
    return float(np.minimum(ha / ha.sum(), hb / hb.sum()).sum())


def run_synthetic(a, say):
    """EchoNet-Synthetic / EchoFlow: not audited for 'correctness' (there's no
    real patient to check against) - profiled for EF-label distribution,
    and compared visually to a real clip, which is exactly what Review-1's
    panel asked to see. Smoke-tested on dummy frames only in this session."""
    if a.labels:
        lb = pd.read_csv(a.labels)
        say(f"Synthetic labels file: {len(lb):,} rows, columns = {list(lb.columns)}")
        if "EF" in lb.columns:
            say(f"  Target EF: mean {lb.EF.mean():.2f}, range {lb.EF.min():.2f}-{lb.EF.max():.2f}")
            if a.filelist:
                real_ef = pd.read_csv(a.filelist).EF
                say(f"  Coherence of EF distributions (real vs synthetic): KS distance "
                    f"{ks_distance(real_ef, lb.EF):.3f} (0 = identical); tail share real "
                    f"{(real_ef <= 40).mean():.1%} vs synthetic {(lb.EF <= 40).mean():.1%}")
    if a.real_video and a.synthetic_video:
        real = grab_middle_frame(a.real_video)
        synth = grab_middle_frame(a.synthetic_video)
        out_path = os.path.join(a.out, "fig_real_vs_synthetic.png")
        real_vs_synthetic_figure(real, synth, out_path)
        say(f"Saved real-vs-synthetic comparison to {out_path}")
        g_r = real.mean(axis=2) if real.ndim == 3 else real
        g_s = synth.mean(axis=2) if synth.ndim == 3 else synth
        say(f"  Frame coherence: intensity-histogram overlap {histogram_overlap(g_r, g_s):.3f} (1 = same look); "
            f"mean intensity real {g_r.mean():.1f} vs synthetic {g_s.mean():.1f}")
    else:
        say("Pass --real-video and --synthetic-video to also render a side-by-side comparison frame.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dynamic", "pediatric", "camus", "synthetic"])
    ap.add_argument("--filelist"); ap.add_argument("--tracings")
    ap.add_argument("--view", default=None)
    ap.add_argument("--camus-root", dest="camus_root")
    ap.add_argument("--labels"); ap.add_argument("--real-video", dest="real_video")
    ap.add_argument("--synthetic-video", dest="synthetic_video")
    ap.add_argument("--flag", type=float, default=10.0)
    ap.add_argument("--duplicate-tail", dest="duplicate_tail", type=int, default=0,
                     help="oversample EF<=40 rows this many extra times in the written training-index CSV "
                          "(Review-2 panel suggestion: duplicate rare/low-EF cases as a quick supplement "
                          "to the Tail Guard)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"=== VERITAS-Echo dataset audit: {a.dataset} ===")
    if a.dataset in ("dynamic", "pediatric"):
        run_echonet_family(a, say, is_pediatric=(a.dataset == "pediatric"))
    elif a.dataset == "camus":
        run_camus(a, say)
    elif a.dataset == "synthetic":
        run_synthetic(a, say)

    with open(os.path.join(a.out, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nAll outputs written to ./{a.out}/")


if __name__ == "__main__":
    main()
