"""
VERITAS-Echo - Outer loop building blocks: the data-side half of one retraining
generation, built so each panel concern is enforced in code, not just promised.

  contamination_mixer()   Review-2: "you should never go for [fully] synthetic."
                          The synthetic share is capped (MAX_SYNTHETIC_RATIO) and the
                          schedule starts small (5%), so a generation can never be
                          mostly or entirely generated data.            (FR-OL-2)
  assert_no_test_leak()   The locked real test set can never enter a training
                          index. Raises instead of warning.             (FR-OL-8, NFR-DATA-2)
  tail_guard()            Review-1/2: rare low-EF cases must survive filtering. After
                          the verifiers have removed whatever they remove, low-EF
                          videos are topped back up to a minimum share.  (FR-OL-4)
  tail_replay_schedule()  Review-2: "duplicate the edge cases and give them to the
                          model every now and then so the parameters aren't lost."
                          Produces a training order in which a tail video recurs
                          after every K ordinary videos, rather than all duplicates
                          sitting in one block that the model sees once.
  synthetic_schedule()    Review-1: the 8-generation figure is a provisional default.
                          The schedule is a parameter, not a constant.

Not in this file yet (Phase 2): the learned verifiers V1-V3, the LLM auditor V5, the
ensemble V6, retraining itself, and the Collapse Observatory dashboard. Verifier V4
(physics) is veritas_inner_loop.physics_check(), reused as-is.

Usage
  python veritas_outer_loop.py --selftest
  python veritas_outer_loop.py --filelist FileList.csv --synthetic-labels synth.csv \
         --ratio 0.10 --tail-min-share 0.15 --replay-every 20 --out results_outer
"""
import argparse
import os

import numpy as np
import pandas as pd

MAX_SYNTHETIC_RATIO = 0.75   # hard ceiling: real data is always at least a quarter of a generation
TAIL_EF = 40.0               # EF <= 40% is the tail the Tail Guard protects


def synthetic_schedule(n_generations=8, start=0.05, end=MAX_SYNTHETIC_RATIO):
    """Generation 0 is real-only; generations 1..n-1 sweep from `start` to `end`.
    n_generations=8 is the provisional default from Review-1, to be revised once
    baseline results show where collapse actually begins."""
    if end > MAX_SYNTHETIC_RATIO:
        raise ValueError(f"end ratio {end} exceeds the cap of {MAX_SYNTHETIC_RATIO}")
    if n_generations < 2:
        return [0.0]
    return [0.0] + list(np.round(np.linspace(start, end, n_generations - 1), 3))


def assert_no_test_leak(train_ids, locked_test_ids):
    leaked = set(train_ids) & set(locked_test_ids)
    if leaked:
        raise AssertionError(f"{len(leaked)} locked test videos found in the training index, "
                             f"e.g. {sorted(leaked)[:3]} - the generation is invalid")


def contamination_mixer(real, synthetic, ratio, n_total=None, seed=0, locked_test_ids=()):
    """real / synthetic: DataFrames with at least FileName and EF.
    Returns a DataFrame of n_total rows with exactly round(ratio*n_total) synthetic rows
    and a `source` column. Never uses a locked test video."""
    if not 0 <= ratio <= MAX_SYNTHETIC_RATIO:
        raise ValueError(f"synthetic ratio {ratio} outside [0, {MAX_SYNTHETIC_RATIO}] - "
                         "a generation may never be mostly or wholly synthetic")
    real = real[~real.FileName.isin(set(locked_test_ids))]
    n_total = n_total or len(real)
    n_syn = int(round(ratio * n_total))
    n_real = n_total - n_syn
    if n_real > len(real) or n_syn > len(synthetic):
        raise ValueError(f"not enough data: need {n_real} real / {n_syn} synthetic, "
                         f"have {len(real)} / {len(synthetic)}")
    rng = np.random.default_rng(seed)
    r = real.iloc[rng.choice(len(real), n_real, replace=False)].assign(source="real")
    s = synthetic.iloc[rng.choice(len(synthetic), n_syn, replace=False)].assign(source="synthetic")
    mix = pd.concat([r, s], ignore_index=True)
    assert_no_test_leak(mix.FileName, locked_test_ids)
    return mix


def tail_guard(kept, pool, min_share=0.12, seed=0):
    """kept: what survived the verifier bank. pool: every candidate before filtering.
    If the verifiers left the tail below `min_share`, add tail videos back - first ones
    the verifiers dropped, then duplicates - until the share is restored. Returns the
    corrected set and a short log of what was done (logged for provenance, FR-OL-7)."""
    rng = np.random.default_rng(seed)
    kept = kept.copy()
    is_tail = kept.EF <= TAIL_EF
    share = is_tail.mean() if len(kept) else 0.0
    log = {"tail_share_before": float(share), "reinstated": 0, "duplicated": 0}
    if share >= min_share:
        log["tail_share_after"] = float(share)
        return kept, log
    # need t tail rows such that (tail+t)/(n+t) >= min_share
    n, tail_n = len(kept), int(is_tail.sum())
    need = int(np.ceil((min_share * n - tail_n) / (1 - min_share)))
    dropped_tail = pool[(pool.EF <= TAIL_EF) & ~pool.FileName.isin(kept.FileName)]
    add = dropped_tail.iloc[:need]
    log["reinstated"] = len(add)
    rest = need - len(add)
    if rest > 0:
        tail_rows = kept[is_tail] if tail_n else pool[pool.EF <= TAIL_EF]
        if len(tail_rows) == 0:
            raise ValueError("no low-EF videos exist anywhere in the pool; the tail cannot be protected")
        dup = tail_rows.iloc[rng.choice(len(tail_rows), rest, replace=True)]
        add = pd.concat([add, dup])
        log["duplicated"] = rest
    kept = pd.concat([kept, add], ignore_index=True)
    log["tail_share_after"] = float((kept.EF <= TAIL_EF).mean())
    return kept, log


def tail_replay_schedule(train, every=20, seed=0):
    """Order the training set so a tail video is replayed after every `every` ordinary
    videos (cycling through the tail), on top of each tail video's own normal slot."""
    rng = np.random.default_rng(seed)
    tail = train[train.EF <= TAIL_EF].reset_index(drop=True)
    rest = train[train.EF > TAIL_EF].sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(tail) == 0 or every <= 0:
        return train.sample(frac=1, random_state=seed).reset_index(drop=True)
    order, t = [], 0
    for i, row in enumerate(rest.itertuples(index=False)):
        order.append(row._asdict() | {"replay": False})
        if (i + 1) % every == 0:
            order.append(tail.iloc[t % len(tail)].to_dict() | {"replay": True})
            t += 1
    order += [r | {"replay": False} for r in tail.sample(frac=1, random_state=seed).to_dict("records")]
    return pd.DataFrame(order)


# --------------------------------------------------------------------- self-test
def selftest():
    rng = np.random.default_rng(1)
    real = pd.DataFrame({"FileName": [f"R{i}" for i in range(1000)],
                         "EF": np.clip(rng.normal(58, 11, 1000), 8, 85)})
    synth = pd.DataFrame({"FileName": [f"S{i}" for i in range(1000)],
                          "EF": np.clip(rng.normal(60, 8, 1000), 8, 85)})
    test_ids = set(real.FileName[:200])

    print("schedule:", synthetic_schedule())
    try:
        contamination_mixer(real, synth, 1.0); raise SystemExit("FAIL: fully synthetic generation allowed")
    except ValueError as e:
        print("fully synthetic rejected:", str(e)[:60], "...")

    mix = contamination_mixer(real, synth, 0.25, n_total=400, locked_test_ids=test_ids)
    frac = (mix.source == "synthetic").mean()
    print(f"mixer ratio 0.25 -> {frac:.3f} synthetic, n={len(mix)}")
    assert abs(frac - 0.25) <= 0.01
    assert not set(mix.FileName) & test_ids

    try:
        assert_no_test_leak(["R5", "R900"], test_ids); raise SystemExit("FAIL: leak not caught")
    except AssertionError as e:
        print("leak caught:", str(e)[:50], "...")

    # simulate verifiers that are biased against the tail: drop 90% of low-EF videos
    drop = (mix.EF <= TAIL_EF) & (rng.random(len(mix)) < 0.9)
    kept = mix[~drop]
    guarded, log = tail_guard(kept, mix, min_share=0.12)
    print("tail guard:", log)
    assert log["tail_share_after"] >= 0.12

    sched = tail_replay_schedule(guarded, every=20)
    gaps = np.diff(np.flatnonzero(sched.replay.values))
    print(f"replay schedule: {int(sched.replay.sum())} replays, spacing {gaps.min() if len(gaps) else '-'}-"
          f"{gaps.max() if len(gaps) else '-'} rows")
    assert sched.replay.sum() > 0 and (len(gaps) == 0 or gaps.max() <= 21)
    print("All outer-loop self-tests passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--filelist"); ap.add_argument("--synthetic-labels", dest="synth")
    ap.add_argument("--ratio", type=float, default=0.05)
    ap.add_argument("--tail-min-share", dest="tail_min", type=float, default=0.12)
    ap.add_argument("--replay-every", dest="every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_outer")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not (a.filelist and a.synth):
        ap.error("--filelist and --synthetic-labels are required unless --selftest")
    os.makedirs(a.out, exist_ok=True)
    fl = pd.read_csv(a.filelist)
    test_ids = set(fl.FileName[fl.Split.astype(str).str.upper() == "TEST"]) if "Split" in fl else set()
    real = fl[~fl.FileName.isin(test_ids)][["FileName", "EF"]]
    synth = pd.read_csv(a.synth)[["FileName", "EF"]]
    n_total = min(len(real), int(len(synth) / max(a.ratio, 1e-9))) if a.ratio > 0 else len(real)
    mix = contamination_mixer(real, synth, a.ratio, n_total=n_total, seed=a.seed, locked_test_ids=test_ids)
    guarded, log = tail_guard(mix, mix, a.tail_min, seed=a.seed)
    sched = tail_replay_schedule(guarded, a.every, seed=a.seed)
    sched.to_csv(os.path.join(a.out, "training_schedule.csv"), index=False)
    print(f"generation mix: {len(mix)} videos, {(mix.source == 'synthetic').mean():.1%} synthetic; "
          f"locked test videos excluded: {len(test_ids)}")
    print("tail guard:", log)
    print(f"schedule written: {len(sched)} rows, {int(sched.replay.sum())} tail replays "
          f"-> {os.path.join(a.out, 'training_schedule.csv')}")


if __name__ == "__main__":
    main()
