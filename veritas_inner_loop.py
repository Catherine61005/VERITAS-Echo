"""
VERITAS-Echo - Inner loop, end to end: video -> mask -> volume curve -> EF ->
physics check -> retry -> explainable report.

This is Phase-2's first real build, not another dataset audit. It takes the
exact, already-validated volume-from-mask code (mask_to_volume_by_disks,
proven against CAMUS's ground truth and against EchoNet-Dynamic's chord
tracings in veritas_dataset_audit.py) and wires it to an actual per-frame
segmenter, a physics-plausibility check with thresholds calibrated from real
data instead of guessed, and a heuristic version of the Clinical Auditor
Agent's retry loop.

HONEST STATUS - read this before presenting any number from this file
  - mask_to_volume_by_disks(): fully validated (see veritas_dataset_audit.py
    and its ~6% check against an analytic ellipsoid volume).
  - physics_check(): the dV/dt and periodicity math is implemented and
    smoke-tested, but calibrate_thresholds() needs to be run against your
    real VolumeTracings.csv to get real kappa_max / rho_min numbers - the
    defaults here are placeholders, not calibrated values.
  - segment_frame(backend="classical"): this is NOT SAM-2. It is a plain
    Otsu-threshold + largest-contour segmenter, included ONLY so the full
    pipeline can be run and demoed end to end today without a GPU, a
    multi-gigabyte checkpoint download, and a torch install (none of which
    completed in the session this file was written in - see
    segment_frame(backend="sam2") below for exactly what to change once
    those are available). Do not present classical-backend EF numbers as
    SAM-2 results.
  - segment_frame(backend="sam2"): a real integration point, not a working
    implementation - it raises NotImplementedError with the three lines you
    need to fill in once `pip install torch sam2` and a checkpoint are
    available. Swapping it in requires no other change to this file.
  - clinical_auditor_retry(): a heuristic stand-in (contrast stretch + a
    tighter crop around the previous mask) for what will eventually be an
    LLM-driven re-prompt. It is deliberately simple so the LOOP mechanics
    (detect failure -> retry -> re-check) can be proven before the smarter
    agent is built.
  - Everything in this file has been smoke-tested on a synthetically
    generated pulsing-blob video (see the bottom of this docstring for the
    command), not on a real echo video, because none was available in this
    session.

Usage
  # one video, classical backend (no GPU / checkpoint needed):
  python veritas_inner_loop.py --video clip.avi --backend classical --out results_inner

  # once SAM-2 is set up (see segment_frame below):
  python veritas_inner_loop.py --video clip.avi --backend sam2 --checkpoint sam2_hiera_small.pt --out results_inner

  # calibrate physics thresholds from real EchoNet-Dynamic tracings before trusting physics_check():
  python veritas_inner_loop.py --calibrate --tracings VolumeTracings.csv --filelist FileList.csv --fps 50

  # self-test on a synthetic pulsing-blob video (no dataset needed):
  python veritas_inner_loop.py --selftest
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

NAVY, TEAL, ORANGE, RED = "#252C5C", "#1FA9A9", "#E48312", "#C0392B"


# --------------------------------------------------------------------- Step 1: perception
def segment_frame(frame_rgb, backend="classical", checkpoint=None, prev_mask=None):
    """Return a boolean LV mask for one frame. `backend` selects the implementation.

    classical: Otsu threshold + largest connected component. A placeholder,
    not a left-ventricle segmenter - it will happily segment any bright
    blob. Good enough to prove the pipeline mechanics; not a measurement
    of real segmentation accuracy.

    sam2: wire in the real model here. Not implemented in this file because
    this session had neither a GPU-ready torch install nor a downloaded
    SAM-2/MedSAM2 checkpoint; the three lines to add are in the comment
    below so this is a 10-minute edit once those are available, not a rewrite.
    """
    if backend == "classical":
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if prev_mask is not None:
            # Clinical-Auditor-style focus: restrict thresholding to a padded
            # box around the previous frame's mask, which is a cheap way to
            # resist background clutter stealing the segmentation.
            ys, xs = np.nonzero(prev_mask)
            if len(ys):
                pad = 15
                y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, gray.shape[0])
                x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, gray.shape[1])
                roi = np.zeros_like(gray, dtype=bool)
                sub = gray[y0:y1, x0:x1]
                _, th = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                roi[y0:y1, x0:x1] = th > 0
                mask = roi
            else:
                _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                mask = th > 0
        else:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = th > 0
        mask = _largest_component(mask)
        return mask

    if backend == "sam2":
        # To wire in the real model:
        #   from sam2.build_sam import build_sam2
        #   from sam2.sam2_image_predictor import SAM2ImagePredictor
        #   predictor = SAM2ImagePredictor(build_sam2(config, checkpoint))
        #   predictor.set_image(frame_rgb)
        #   mask, _, _ = predictor.predict(point_coords=..., point_labels=...)
        # A point or box prompt near the LV centre is still needed on the
        # first frame (or from the previous frame's mask centroid on later
        # frames) - SAM-2 is promptable, not a push-button LV detector.
        raise NotImplementedError(
            "SAM-2 backend not wired in this session (no torch/checkpoint available). "
            "See the comment above this line for the integration point."
        )
    raise ValueError(f"unknown backend: {backend}")


def _largest_component(mask):
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if num <= 1:
        return mask
    sizes = [(labels == i).sum() for i in range(1, num)]
    biggest = 1 + int(np.argmax(sizes))
    return labels == biggest


# --------------------------------------------------------------------- Step 1b: the validated volume code
def mask_to_volume_by_disks(mask, n_disks=20):
    """Identical to the function of the same name in veritas_dataset_audit.py,
    validated there against an analytic ellipsoid volume (~6% error, the
    expected pixel-discretization error for a 20-band estimate) and used on
    CAMUS's real ground-truth masks."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0
    y0, y1 = ys.min(), ys.max()
    L = float(y1 - y0)
    if L == 0:
        return 0.0
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


# --------------------------------------------------------------------- Step 2: physics check
DEFAULT_KAPPA_MAX = 0.35   # fraction of V_ED per frame - PLACEHOLDER, calibrate with --calibrate
DEFAULT_RHO_MIN = 0.5      # autocorrelation-based periodicity score, 0-1 - PLACEHOLDER


# --------------------------------------------------------------------- Review-1: preprocessing
def preprocess_frames(frames, size=112):
    """Review-1 asked how video is extracted and what preprocessing is applied.
    Every frame is converted to grayscale (echo is single-channel; colour
    overlays and ECG traces carry no LV information), resized to the
    112x112 resolution EchoNet-Dynamic/-Pediatric ship at, and intensity-
    normalised per video (min-max over the whole clip, not per frame, so a
    genuine brightness change between frames is preserved). Returned as
    3-channel uint8 so any segmentation backend can consume it."""
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f for f in frames]
    gray = [cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA) for g in gray]
    stack = np.stack(gray).astype(np.float32)
    lo, hi = np.percentile(stack, 1), np.percentile(stack, 99)
    stack = np.clip((stack - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return [cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2RGB) for g in stack]


# --------------------------------------------------------------------- Review-2: heart rate by demographic
# Review-2: an athlete's heart can beat 40-45 times a minute, an inactive adult's
# 70-75, and a child's far faster. These are PLAUSIBILITY bounds (could this be a
# real heartbeat at all?), deliberately wide so an athlete or an infant is never
# rejected for being normal-for-them. They are not diagnostic normal ranges.
HR_PLAUSIBLE_BPM = [  # (max_age_exclusive, low, high)
    (1, 70, 230),     # infants
    (18, 45, 220),    # children and adolescents
    (200, 30, 200),   # adults, including trained athletes with resting rates near 40
]


def hr_bounds_for_age(age):
    if age is None:
        return 30, 230  # unknown age: accept anything physiologically possible
    for max_age, lo, hi in HR_PLAUSIBLE_BPM:
        if age < max_age:
            return lo, hi
    return HR_PLAUSIBLE_BPM[-1][1:]


def estimate_heart_rate(volume_curve, fps):
    """Heart rate from the volume curve itself: the lag of the first strong
    autocorrelation peak is one cardiac cycle. Returns bpm, or None if the
    clip is too short to contain a full cycle (common for 2-second clips)."""
    v = np.asarray(volume_curve, dtype=float)
    if fps is None or len(v) < 6 or v.std() == 0:
        return None
    v = v - v.mean()
    ac = np.correlate(v, v, mode="full")[len(v) - 1:]
    ac = ac / ac[0]
    for lag in range(2, len(ac) - 1):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.2:
            return 60.0 * fps / lag
    return None


# --------------------------------------------------------------------- Review-2: EF interpretation by sex
# ASE/EACVI 2015 chamber-quantification guideline (Lang et al., JASE 2015):
# normal 2D LVEF is 52-72% in men and 54-74% in women; 41-51% (men) / 41-53%
# (women) is mildly abnormal, 30-40% moderately abnormal, <30% severely abnormal.
# These are ADULT reference ranges; they are not applied to children.
ASE_2015_EF = {"M": (52, 72, 51), "F": (54, 74, 53)}  # (normal_low, normal_high, mild_high)


def interpret_ef(ef, sex=None, age=None):
    """Review-2 asked whether there is a specific EF value that counts as working
    properly for a given patient. Returns a category using sex-specific adult
    cut-offs when sex is known, and says so explicitly when they don't apply."""
    if ef is None or np.isnan(ef):
        return "not interpretable"
    if age is not None and age < 18:
        return ("paediatric patient: adult ASE cut-offs not applied; compare against the "
                "EchoNet-Pediatric distribution for this age band")
    s = (sex or "").strip().upper()[:1]
    if s not in ASE_2015_EF:
        lo, hi, mild = 52, 72, 51  # sex unknown: use the more conservative (male) lower limit
        note = " (sex unknown, male cut-off used)"
    else:
        lo, hi, mild = ASE_2015_EF[s]
        note = f" (ASE 2015, {'male' if s == 'M' else 'female'} range {lo}-{hi}%)"
    if ef > hi:
        cat = "above normal range"
    elif ef >= lo:
        cat = "normal"
    elif ef > 40:
        cat = "mildly abnormal"
    elif ef >= 30:
        cat = "moderately abnormal"
    else:
        cat = "severely abnormal"
    return cat + note


def periodicity_score(volume_curve):
    """A simple periodicity measure: normalized autocorrelation at the lag
    that best matches a repeating heartbeat, found by looking for the
    strongest peak after lag 0. Returns 0-1; 1 = perfectly repeating."""
    v = np.asarray(volume_curve, dtype=float)
    if len(v) < 6 or v.std() == 0:
        return 0.0
    v = v - v.mean()
    ac = np.correlate(v, v, mode="full")[len(v) - 1:]
    ac = ac / ac[0]
    # ignore the first couple of lags (trivially high autocorrelation)
    peak = ac[2:].max() if len(ac) > 2 else 0.0
    return float(np.clip(peak, 0.0, 1.0))


def physics_check(volume_curve, kappa_max=DEFAULT_KAPPA_MAX, rho_min=DEFAULT_RHO_MIN,
                   label_ef=None, delta=10.0, fps=None, age=None):
    """Implements the four rules from the Requirement Specification (FR-IL-5):
    |dV/dt| <= kappa_max, EDV > ESV > 0, periodicity >= rho_min, and, for
    synthetic videos with a target label, |EF - label| <= delta.
    Returns (passed: bool, reasons: list[str], info: dict)."""
    v = np.asarray(volume_curve, dtype=float)
    reasons = []
    v_ed, v_es = v.max(), v.min()

    if not (v_ed > v_es > 0):
        reasons.append(f"EDV>ESV>0 failed: V_ED={v_ed:.1f}, V_ES={v_es:.1f}")

    dv = np.diff(v)
    max_rate = np.max(np.abs(dv)) / v_ed if v_ed > 0 else np.inf
    if max_rate > kappa_max:
        reasons.append(f"|dV/dt| limit exceeded: {max_rate:.2f} > kappa_max={kappa_max:.2f}")

    rho = periodicity_score(v)
    if rho < rho_min:
        reasons.append(f"periodicity too low: {rho:.2f} < rho_min={rho_min:.2f}")

    hr = estimate_heart_rate(v, fps)
    if hr is not None:
        lo, hi = hr_bounds_for_age(age)
        if not (lo <= hr <= hi):
            reasons.append(f"heart rate {hr:.0f} bpm outside plausible range {lo}-{hi} bpm for age {age}")

    ef = (v_ed - v_es) / v_ed * 100 if v_ed > 0 else float("nan")
    if label_ef is not None and abs(ef - label_ef) > delta:
        reasons.append(f"|EF-label| exceeded: |{ef:.1f}-{label_ef:.1f}| > {delta:g}")

    info = {"V_ED": v_ed, "V_ES": v_es, "EF": ef, "max_dVdt_rate": max_rate, "periodicity": rho,
            "heart_rate_bpm": hr}
    return (len(reasons) == 0), reasons, info


def calibrate_thresholds(tracings_csv, filelist_csv, fps_default=50.0, percentile=99.0):
    """Derive kappa_max and rho_min empirically from real EchoNet-Dynamic
    tracings instead of guessing: compute the per-frame volume curve for
    every video that has >2 traced frames (most EchoNet videos only have 2 -
    ED and ES - so this works best on a dataset or subset with denser
    per-frame tracing; with only 2 points per video it falls back to a
    single dV/dt estimate per video, which still gives a usable distribution
    across many videos)."""
    vt = pd.read_csv(tracings_csv)
    name_col = "FileName" if "FileName" in vt.columns else vt.columns[0]
    vt[name_col] = vt[name_col].astype(str).str.replace(".avi", "", regex=False)
    rates, periodicities = [], []
    for f, g in vt.groupby(name_col):
        frames = sorted(g.Frame.unique())
        vols, used_frames = [], []
        for fr in frames:
            rows = g[g.Frame == fr][["X1", "Y1", "X2", "Y2"]].values[:21]
            if len(rows) < 2:
                continue
            long_axis = np.hypot(rows[0, 2] - rows[0, 0], rows[0, 3] - rows[0, 1])
            d = np.hypot(rows[1:, 2] - rows[1:, 0], rows[1:, 3] - rows[1:, 1])
            vols.append(np.pi / 4 * np.sum(d ** 2) * long_axis / len(d))
            used_frames.append(fr)
        if len(vols) >= 2:
            v = np.array(vols)
            # Traced frames are usually ED and ES, many frames apart. Dividing by the frame gap
            # gives the AVERAGE per-frame change between them; the true peak per-frame change is
            # at least this large, so a percentile of it is a lower bound for kappa_max.
            gaps = np.maximum(np.diff(np.array(used_frames, dtype=float)), 1.0)
            rates.append(np.max(np.abs(np.diff(v)) / gaps) / v.max())
        if len(vols) >= 6:
            periodicities.append(periodicity_score(vols))
    kappa_max = float(np.percentile(rates, percentile)) if rates else DEFAULT_KAPPA_MAX
    rho_min = float(np.percentile(periodicities, 100 - percentile)) if periodicities else DEFAULT_RHO_MIN
    print(f"Calibrated from {len(rates)} videos (ED/ES-only dV/dt) and {len(periodicities)} "
          f"videos with >=6 traced frames (periodicity):")
    print(f"  kappa_max lower bound (P{percentile:g} of the average per-frame change between "
          f"traced frames) = {kappa_max:.3f} of EDV per frame")
    print(f"  rho_min   (P{100 - percentile:g}) = {rho_min:.3f}")
    print("  NOTE: most EchoNet-Dynamic videos only have 2 traced frames (ED, ES), so the "
          "periodicity estimate above is likely thin - treat rho_min as a starting point, "
          "and prefer calibrating it from full-video SAM-2/MedSAM2 volume curves once those exist.")
    return kappa_max, rho_min


# --------------------------------------------------------------------- Step 4: heuristic retry
def clinical_auditor_retry(frame_rgb, prev_mask, attempt):
    """Heuristic stand-in for the LLM-driven Clinical Auditor Agent (FR-IL-6).
    Tries a different, cheap fix on each retry: attempt 1 = contrast
    stretch, attempt 2 = tighter crop around the previous mask's centroid.
    Returns an adjusted frame to re-segment; does not touch the mask itself."""
    if attempt == 1:
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    if attempt == 2 and prev_mask is not None and prev_mask.any():
        ys, xs = np.nonzero(prev_mask)
        cy, cx = int(ys.mean()), int(xs.mean())
        h, w = frame_rgb.shape[:2]
        half = max(h, w) // 3
        y0, y1 = max(cy - half, 0), min(cy + half, h)
        x0, x1 = max(cx - half, 0), min(cx + half, w)
        out = np.zeros_like(frame_rgb)
        out[y0:y1, x0:x1] = frame_rgb[y0:y1, x0:x1]
        return out
    return frame_rgb


# --------------------------------------------------------------------- Step 3: the one-shot pipeline + loop
def run_inner_loop_on_video(path, backend="classical", checkpoint=None, max_retries=2,
                             kappa_max=DEFAULT_KAPPA_MAX, rho_min=DEFAULT_RHO_MIN, label_ef=None,
                             fps=None, sex=None, age=None, preprocess=True):
    """FR-IL-1 through FR-IL-9 end to end on one video. Returns a report dict;
    never raises on a physics failure - it retries, then reports honestly if
    it still fails after max_retries, rather than silently releasing a
    bad EF (FR-IL-8)."""
    cap = cv2.VideoCapture(path)
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or None
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[:, :, ::-1])
    cap.release()
    if not frames:
        return {"ok": False, "error": f"no frames read from {path}"}
    if preprocess:
        frames = preprocess_frames(frames)

    attempt = 0
    prev_mask = None
    report = {"video": os.path.basename(path), "n_frames": len(frames)}
    while True:
        volumes, masks = [], []
        for fr in frames:
            mask = segment_frame(fr, backend=backend, checkpoint=checkpoint, prev_mask=prev_mask)
            masks.append(mask)
            volumes.append(mask_to_volume_by_disks(mask))
            prev_mask = mask
        passed, reasons, info = physics_check(volumes, kappa_max=kappa_max, rho_min=rho_min,
                                              label_ef=label_ef, fps=fps, age=age)
        report[f"attempt_{attempt}"] = {"passed": passed, "reasons": reasons, **info}
        if passed or attempt >= max_retries:
            report["final"] = {"passed": passed, "reasons": reasons, **info, "attempts_used": attempt + 1}
            break
        # FR-IL-6: retry on the frame that drove the worst-looking part of the curve
        worst_frame_idx = int(np.argmax(np.abs(np.diff(volumes)))) if len(volumes) > 1 else 0
        frames[worst_frame_idx] = clinical_auditor_retry(frames[worst_frame_idx], prev_mask, attempt + 1)
        attempt += 1
    ef = report["final"]["EF"]
    report["final"]["ef_band"] = ("EF <= 40% (tail)" if ef <= 40 else "40-55%" if ef <= 55 else "> 55%")
    report["final"]["is_tail"] = bool(ef <= 40)
    report["final"]["demographics"] = {"sex": sex, "age": age, "fps": fps}
    # FR-IL-8: only a verified EF gets a clinical-style interpretation
    report["final"]["interpretation"] = (interpret_ef(ef, sex, age) if report["final"]["passed"]
                                         else "withheld: EF failed the physics check")
    report["ok"] = True
    return report


# --------------------------------------------------------------------- self-test
def make_synthetic_pulsing_video(path, n_frames=30, size=160, good=True, amp=None, cycles=2.3, fps=20):
    """Writes a small synthetic video of a bright blob that grows/shrinks
    like a very simplified heartbeat, purely so this file's pipeline can be
    exercised end to end without any real echo data."""
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    vw = cv2.VideoWriter(path, fourcc, fps, (size, size))
    cy, cx = size // 2, size // 2
    for i in range(n_frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        phase = i / n_frames * 2 * np.pi * cycles
        r = 35 + (amp if amp is not None else (10 if good else 25)) * np.sin(phase)
        cv2.circle(frame, (cx, cy), max(int(r), 5), (200, 200, 200), -1)
        noise = (np.random.default_rng(i).normal(0, 5, frame.shape)).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        vw.write(frame[:, :, ::-1])
    vw.release()


def selftest():
    import tempfile
    d = tempfile.mkdtemp()
    good_path = os.path.join(d, "good.avi")
    make_synthetic_pulsing_video(good_path, good=True)
    print("--- self-test: well-behaved synthetic video ---")
    r = run_inner_loop_on_video(good_path, backend="classical", max_retries=2)
    print(r["final"])
    assert r["ok"]

    print("--- self-test: weak heart (low EF, small motion) must pass, not be rejected ---")
    weak_path = os.path.join(d, "weak.avi")
    make_synthetic_pulsing_video(weak_path, amp=3)
    rw = run_inner_loop_on_video(weak_path, backend="classical", max_retries=2)
    print(rw["final"])
    assert rw["final"]["is_tail"], "expected the weak-heart video to land in the EF <= 40% band"
    assert rw["final"]["passed"], "physics_check rejected a valid low-EF heart: the tail would be filtered out"

    print("--- self-test: athlete's slow heart (~45 bpm) must pass the heart-rate rule ---")
    ath_path = os.path.join(d, "athlete.avi")
    make_synthetic_pulsing_video(ath_path, n_frames=80, cycles=3, fps=20)  # 80 frames/3 beats at 20 fps = 45 bpm
    ra = run_inner_loop_on_video(ath_path, backend="classical", max_retries=2, age=25, sex="M")
    print({k: ra["final"][k] for k in ("passed", "heart_rate_bpm", "reasons")})
    assert ra["final"]["heart_rate_bpm"] is not None and 38 <= ra["final"]["heart_rate_bpm"] <= 52, \
        "test video did not come out near 45 bpm, so this test would not be testing a slow heart"
    assert ra["final"]["passed"], "an athlete-range heart rate was rejected"

    print("--- self-test: heart rate impossible for an adult must fail ---")
    v = 100 + 30 * np.sin(np.arange(40) * 2 * np.pi / 4)  # one beat every 4 frames
    ok_, reasons_, info_ = physics_check(v, fps=50, age=40)  # 4 frames @50fps = 750 bpm
    print("hr:", round(info_["heart_rate_bpm"]), "passed:", ok_)
    assert not ok_ and any("heart rate" in r for r in reasons_)

    print("--- self-test: sex-specific EF interpretation (ASE 2015) ---")
    m, f = interpret_ef(53, "M"), interpret_ef(53, "F")
    print("EF 53% male:", m, "| female:", f)
    assert m.startswith("normal") and f.startswith("mildly abnormal")
    assert interpret_ef(53, "F", age=8).startswith("paediatric")

    print("--- self-test: physics_check unit check (known-bad curve must fail) ---")
    bad_curve = [100, 5, 95, 3, 98]  # implausible, near-total emptying every other frame
    passed, reasons, info = physics_check(bad_curve)
    print("passed:", passed, "reasons:", reasons)
    assert not passed, "physics_check should have rejected an implausible curve"

    print("All self-tests passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--backend", default="classical", choices=["classical", "sam2"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--max-retries", dest="max_retries", type=int, default=2)
    ap.add_argument("--label-ef", dest="label_ef", type=float, default=None)
    ap.add_argument("--sex", default=None, help="M or F, enables ASE 2015 sex-specific EF interpretation")
    ap.add_argument("--age", type=float, default=None, help="years; sets heart-rate bounds, disables adult cut-offs under 18")
    ap.add_argument("--kappa-max", dest="kappa_max", type=float, default=DEFAULT_KAPPA_MAX)
    ap.add_argument("--rho-min", dest="rho_min", type=float, default=DEFAULT_RHO_MIN)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--tracings")
    ap.add_argument("--filelist")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="results_inner")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if a.calibrate:
        calibrate_thresholds(a.tracings, a.filelist, fps_default=a.fps)
        return
    if not a.video:
        ap.error("--video is required unless --selftest or --calibrate is given")

    os.makedirs(a.out, exist_ok=True)
    report = run_inner_loop_on_video(a.video, backend=a.backend, checkpoint=a.checkpoint,
                                      max_retries=a.max_retries, kappa_max=a.kappa_max,
                                      rho_min=a.rho_min, label_ef=a.label_ef, sex=a.sex, age=a.age)
    import json
    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(json.dumps(report.get("final", report), indent=2, default=float))
    print(f"\nFull report written to {os.path.join(a.out, 'report.json')}")


if __name__ == "__main__":
    main()
