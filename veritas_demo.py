"""
VERITAS-Echo - one-command demonstration for reviews (Review-2: "do you have any
demonstration?"). Runs each built piece in order and prints a clear header for
each, so it can be shown live or screen-recorded as a short demo video.

  python veritas_demo.py                                   # self-tests only, no dataset needed
  python veritas_demo.py --filelist FileList.csv --tracings VolumeTracings.csv [--video clip.avi]

Steps
  1. Inner loop self-test: physics check, weak-heart (tail) case, athlete heart rate,
     sex-specific EF interpretation, retry loop.
  2. Outer loop self-test: synthetic cap, test-set leak guard, Tail Guard, tail replay.
  3. (with data) Dataset audit on real EchoNet-Dynamic: attribute relevance, EF validation
     against expert tracings, flagged label errors.
  4. (with data) Outer-loop generation plan on the real FileList at 5% synthetic.
  5. (with --video) Inner loop on one real echo clip.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def step(n, title, cmd):
    print("\n" + "=" * 78)
    print(f"STEP {n}: {title}")
    print("=" * 78, flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"[step {n} exited with code {r.returncode}]")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist"); ap.add_argument("--tracings"); ap.add_argument("--video")
    ap.add_argument("--synthetic-labels", dest="synth")
    ap.add_argument("--out", default="demo_results")
    a = ap.parse_args()
    ok = []
    ok.append(step(1, "Inner loop self-test", ["veritas_inner_loop.py", "--selftest"]))
    ok.append(step(2, "Outer loop self-test", ["veritas_outer_loop.py", "--selftest"]))
    if a.filelist and a.tracings:
        ok.append(step(3, "Dataset audit on real EchoNet-Dynamic", [
            "veritas_dataset_audit.py", "--dataset", "dynamic", "--filelist", a.filelist,
            "--tracings", a.tracings, "--out", os.path.join(a.out, "audit")]))
        if a.synth:
            ok.append(step(4, "Outer-loop generation plan (5% synthetic)", [
                "veritas_outer_loop.py", "--filelist", a.filelist, "--synthetic-labels", a.synth,
                "--ratio", "0.05", "--out", os.path.join(a.out, "outer")]))
    if a.video:
        ok.append(step(5, "Inner loop on a real clip", [
            "veritas_inner_loop.py", "--video", a.video, "--out", os.path.join(a.out, "inner")]))
    print("\n" + "=" * 78)
    print(f"DEMO COMPLETE: {sum(ok)}/{len(ok)} steps succeeded")
    print("=" * 78)


if __name__ == "__main__":
    main()
