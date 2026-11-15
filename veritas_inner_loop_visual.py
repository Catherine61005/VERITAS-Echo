"""
VERITAS-Echo - visual demo of the inner loop, for showing to a panel.

Runs the inner loop on one video and writes an animated GIF (plays in any browser,
PowerPoint or Windows Photos) showing, frame by frame:
  left  - the video frame with the left-ventricle outline the loop found
  right - the volume curve drawing itself, with ED/ES marked and the final verdict

Usage
  pip install numpy opencv-python matplotlib pillow
  python veritas_inner_loop_visual.py --video sample.avi --sex F --age 54
  -> writes sample_inner_loop.gif next to the video
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

import veritas_inner_loop as il

NAVY, TEAL, ORANGE, RED = "#252C5C", "#1FA9A9", "#E48312", "#C0392B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--sex", default=None)
    ap.add_argument("--age", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # same pipeline as veritas_inner_loop.run_inner_loop_on_video, keeping each mask for display
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[:, :, ::-1])
    cap.release()
    if not frames:
        raise SystemExit(f"could not read any frames from {a.video}")
    frames = il.preprocess_frames(frames)
    masks, vols, prev = [], [], None
    for f in frames:
        m = il.segment_frame(f, prev_mask=prev)
        masks.append(m); vols.append(il.mask_to_volume_by_disks(m)); prev = m
    vols = np.array(vols)
    passed, reasons, info = il.physics_check(vols, fps=fps, age=a.age)
    ef = info["EF"]
    verdict = ("PASS: " + il.interpret_ef(ef, a.sex, a.age)) if passed else ("FAIL: " + "; ".join(reasons))
    hr = info.get("heart_rate_bpm")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=90, gridspec_kw={"width_ratios": [1, 1.4]})
    fig.suptitle("VERITAS-Echo inner loop", color=NAVY, fontsize=13, fontweight="bold")
    img = ax1.imshow(frames[0]); ax1.axis("off"); title1 = ax1.set_title("frame 0", fontsize=10)
    contour_lines = []
    ax2.set_xlim(0, len(vols) - 1); ax2.set_ylim(0, 1.1)
    ax2.set_xlabel("frame"); ax2.set_ylabel("volume / EDV")
    ax2.spines[["top", "right"]].set_visible(False)
    (line,) = ax2.plot([], [], color=TEAL, lw=2)
    ed, es = int(vols.argmax()), int(vols.argmin())
    txt = ax2.text(0.03, 0.06, "", transform=ax2.transAxes, fontsize=9.5, color=NAVY, va="bottom",
                   bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
    vmax = vols.max() if vols.max() > 0 else 1.0

    def draw(i):
        img.set_data(frames[i]); title1.set_text(f"frame {i}: LV outline")
        for c in contour_lines:
            c.remove()
        contour_lines.clear()
        cs = ax1.contour(masks[i].astype(float), levels=[0.5], colors=[ORANGE], linewidths=1.5)
        contour_lines.append(cs)
        line.set_data(np.arange(i + 1), vols[: i + 1] / vmax)
        if i == len(vols) - 1:
            ax2.scatter([ed], [1.0], color=NAVY, zorder=3); ax2.annotate("ED", (ed, 1.0), xytext=(3, 3), textcoords="offset points", color=NAVY)
            ax2.scatter([es], [vols[es] / vmax], color=RED, zorder=3); ax2.annotate("ES", (es, vols[es] / vmax), xytext=(3, -12), textcoords="offset points", color=RED)
            hr_txt = f", {hr:.0f} bpm" if hr else ""
            txt.set_text(f"EF {ef:.1f}%{hr_txt}\n{verdict}")
            txt.set_color(TEAL if passed else RED)
        return [img, line, txt]

    # hold the last frame for a while so the verdict can be read
    order = list(range(len(frames))) + [len(frames) - 1] * 15
    anim = FuncAnimation(fig, lambda k: draw(order[k]), frames=len(order), interval=120, blit=False)
    out = a.out or os.path.splitext(a.video)[0] + "_inner_loop.gif"
    fig.tight_layout()
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"EF {ef:.1f}%  passed={passed}  -> {out}")


if __name__ == "__main__":
    main()
