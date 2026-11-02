"""
VERITAS-Echo · Phase 0 data audit + Simpson's-disk EF check on EchoNet-Dynamic.

What it does
  1. Profiles FileList.csv (splits, EF bands, frame sizes, internal consistency).
  2. Recomputes EF from every expert tracing in VolumeTracings.csv with the
     method of disks (the same formula our Temporal Consistency Agent uses)
     and compares it with the label EF.
  3. Flags videos where tracing-EF and label-EF disagree by more than a threshold.
  4. (optional) With --videos, overlays the tracing on the real ED / ES frames of
     the best-matching and worst-matching videos.

Usage
  pip install pandas numpy matplotlib opencv-python
  python veritas_phase0_audit.py --filelist FileList.csv --tracings VolumeTracings.csv \
         [--videos path/to/EchoNet-Dynamic/Videos] [--out results]

Outputs (in --out)
  summary.txt                 all headline numbers
  simpson_ef_vs_label.csv     one row per video: label EF, our EF, error, flag
  fig_ef_histogram.png        EF distribution by band
  fig_simpson_vs_label.png    our EF vs label EF
  fig_tracing_<file>.png      ED/ES frames with tracing (only with --videos)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY, TEAL, ORANGE, RED = "#252C5C", "#1FA9A9", "#E48312", "#C0392B"
BANDS = [(0, 40, "EF <= 40%", RED), (40, 55, "40-55%", ORANGE), (55, 100, "> 55%", TEAL)]


def disk_volume(rows):
    """rows: (21, 4) array of X1,Y1,X2,Y2. Row 0 = long axis, rows 1..20 = disk chords.
    V = (pi/4) * sum(d_i^2) * (L / N)   (pixel units; EF is a ratio so units cancel)"""
    long_axis = np.hypot(rows[0, 2] - rows[0, 0], rows[0, 3] - rows[0, 1])
    d = np.hypot(rows[1:, 2] - rows[1:, 0], rows[1:, 3] - rows[1:, 1])
    return np.pi / 4 * np.sum(d ** 2) * long_axis / len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", default="FileList.csv")
    ap.add_argument("--tracings", default="VolumeTracings.csv")
    ap.add_argument("--videos", default=None, help="folder with the .avi files (optional)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--flag", type=float, default=10.0, help="flag if |EF_ours - EF_label| > this")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # ---------------------------------------------------------------- FileList
    fl = pd.read_csv(a.filelist)
    say(f"FileList.csv: {len(fl):,} videos, columns = {list(fl.columns)}")
    say(f"Splits: {fl.Split.value_counts().to_dict()}")
    say(f"EF: mean {fl.EF.mean():.2f}, median {fl.EF.median():.2f}, min {fl.EF.min():.2f}, max {fl.EF.max():.2f}")
    for lo, hi, name, _ in BANDS:
        n = ((fl.EF > lo) & (fl.EF <= hi)).sum()
        say(f"  {name:>9}: {n:,} videos ({n / len(fl) * 100:.1f}%)")
    sizes = fl.groupby(["FrameHeight", "FrameWidth"]).size().to_dict()
    say(f"Frame sizes: {sizes}")
    say(f"Frames per video: median {fl.NumberOfFrames.median():.0f}, range {fl.NumberOfFrames.min()}-{fl.NumberOfFrames.max()}")
    ef_from_vol = (fl.EDV - fl.ESV) / fl.EDV * 100
    say(f"Label consistency: max |EF - (EDV-ESV)/EDV*100| = {(ef_from_vol - fl.EF).abs().max():.4f}")
    say(f"ESV >= EDV: {(fl.ESV >= fl.EDV).sum()}   non-positive volumes: {((fl.ESV <= 0) | (fl.EDV <= 0)).sum()}")

    # ---------------------------------------------------------------- Tracings
    vt = pd.read_csv(a.tracings)
    vt["FileName"] = vt.FileName.str.replace(".avi", "", regex=False)
    say()
    say(f"VolumeTracings.csv: {len(vt):,} rows, {vt.FileName.nunique():,} videos")
    per_frame = vt.groupby(["FileName", "Frame"]).size()
    say(f"Traced frames per video: {vt.groupby('FileName').Frame.nunique().value_counts().to_dict()}")
    say(f"Rows per traced frame: {per_frame.value_counts().sort_index().to_dict()}  (21 = 1 long axis + 20 chords; more = duplicate tracing)")
    say(f"Videos in FileList with no tracing: {len(set(fl.FileName) - set(vt.FileName))}")

    vols = []
    for (f, frame), g in vt.groupby(["FileName", "Frame"], sort=False):
        rows = g[["X1", "Y1", "X2", "Y2"]].values[:21]  # keep first tracing if duplicated
        vols.append((f, frame, disk_volume(rows)))
    v = pd.DataFrame(vols, columns=["FileName", "Frame", "V"])
    ed = v.loc[v.groupby("FileName").V.idxmax()].set_index("FileName")
    es = v.loc[v.groupby("FileName").V.idxmin()].set_index("FileName")
    res = pd.DataFrame({"ED_frame": ed.Frame, "ES_frame": es.Frame, "V_ED": ed.V, "V_ES": es.V})
    res["EF_ours"] = (res.V_ED - res.V_ES) / res.V_ED * 100
    res = res.join(fl.set_index("FileName")[["EF", "Split"]].rename(columns={"EF": "EF_label"}), how="inner")
    res["abs_error"] = (res.EF_ours - res.EF_label).abs()
    res["flagged"] = res.abs_error > a.flag
    res.sort_values("abs_error", ascending=False).to_csv(os.path.join(a.out, "simpson_ef_vs_label.csv"))

    say()
    say(f"Simpson's-disk EF vs label EF on {len(res):,} videos:")
    say(f"  MAE {res.abs_error.mean():.2f} EF points, median {res.abs_error.median():.2f}, "
        f"bias {(res.EF_ours - res.EF_label).mean():+.2f}, r = {np.corrcoef(res.EF_ours, res.EF_label)[0, 1]:.3f}")
    for lo, hi, name, _ in BANDS:
        m = (res.EF_label > lo) & (res.EF_label <= hi)
        say(f"  {name:>9}: MAE {res.abs_error[m].mean():.2f}")
    say(f"  Flagged (|error| > {a.flag:g}): {res.flagged.sum()} videos ({res.flagged.mean() * 100:.1f}%)")
    say("  Worst 5: " + ", ".join(f"{i} (label {r.EF_label:.1f}, ours {r.EF_ours:.1f})"
                                   for i, r in res.sort_values("abs_error", ascending=False).head(5).iterrows()))

    # ---------------------------------------------------------------- Figures
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(6.0, 2.9), dpi=220)
    bins = np.arange(5, 100, 2.5)
    for lo, hi, name, c in BANDS:
        sel = fl.EF[(fl.EF > lo) & (fl.EF <= hi)]
        ax.hist(sel, bins=bins, color=c, edgecolor="white", linewidth=0.4,
                label=f"{name}: {len(sel):,} ({len(sel) / len(fl) * 100:.1f}%)")
    ax.set_xlabel("label ejection fraction (%)"); ax.set_ylabel("videos")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title(f"EF distribution, all {len(fl):,} EchoNet-Dynamic videos", fontsize=10.5, color=NAVY, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_ef_histogram.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 2.9), dpi=220)
    ok = ~res.flagged
    ax.scatter(res.EF_label[ok], res.EF_ours[ok], s=3, color=NAVY, alpha=0.25, linewidths=0)
    ax.scatter(res.EF_label[~ok], res.EF_ours[~ok], s=9, color=RED, linewidths=0,
               label=f"off by > {a.flag:g} points ({res.flagged.sum()})")
    ax.plot([0, 100], [0, 100], color=TEAL, lw=1); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("label EF (%)"); ax.set_ylabel("our Simpson's-disk EF (%)")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.text(3, 95, f"MAE {res.abs_error.mean():.2f} EF points   r = {np.corrcoef(res.EF_ours, res.EF_label)[0, 1]:.3f}"
                   f"   n = {len(res):,}", fontsize=9.5, color=NAVY, fontweight="bold", va="top")
    ax.set_title("Our volume code on the real tracings vs the label EF", fontsize=10.5, color=NAVY, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_simpson_vs_label.png")); plt.close(fig)

    # ---------------------------------------------------------------- Optional: overlays on real frames
    if a.videos:
        import cv2
        picks = [res.abs_error.idxmin(), res.abs_error.idxmax()]
        for f in picks:
            path = os.path.join(a.videos, f + ".avi")
            if not os.path.exists(path):
                say(f"(video not found: {path})"); continue
            cap = cv2.VideoCapture(path); frames = []
            while True:
                okf, img = cap.read()
                if not okf: break
                frames.append(img[:, :, ::-1])
            cap.release()
            r = res.loc[f]
            fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.8), dpi=220)
            for ax, fr, name, c in [(axs[0], int(r.ED_frame), "ED", TEAL), (axs[1], int(r.ES_frame), "ES", ORANGE)]:
                if fr < len(frames): ax.imshow(frames[fr])
                rows = vt[(vt.FileName == f) & (vt.Frame == fr)][["X1", "Y1", "X2", "Y2"]].values[:21]
                for x1, y1, x2, y2 in rows[1:]:
                    ax.plot([x1, x2], [y1, y2], color=c, lw=0.5)
                ax.set_title(f"{name} (frame {fr})", fontsize=9, color=c); ax.axis("off")
            fig.suptitle(f"{f}: label EF {r.EF_label:.1f}, ours {r.EF_ours:.1f}", fontsize=9.5, color=NAVY)
            fig.tight_layout(); fig.savefig(os.path.join(a.out, f"fig_tracing_{f}.png")); plt.close(fig)
            say(f"Saved tracing overlay for {f}")

    with open(os.path.join(a.out, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nAll outputs written to ./{a.out}/")


if __name__ == "__main__":
    main()
