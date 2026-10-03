import csv
import os
import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from traces import make_shift_trace, locality_trace
from policies import features, next_use, train_model

NUM_FRAMES = 10
TRAIN_SEED = 1
TEST_SEEDS = range(100, 110)
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def explain(page, recency, frequency, confidence):
    return (f"Evicted page {page}: unused for {recency} steps and requested "
            f"{frequency} time(s) in the last 50 requests. "
            f"Confidence {confidence:.2f}.")


# def learned_with_log(trace, num_frames, model, shift):
#     frames = set()
#     last_used = {}
#     log = []
#     for t, page in enumerate(trace):
#         if page not in frames:
#             if len(frames) == num_frames:
#                 pages = list(frames)
#                 rows = [features(trace, p, t, last_used) for p in pages]
#                 scores = model.predict_proba(rows)[:, 1]
#                 i = int(scores.argmax())
#                 victim = pages[i]
#                 confidence = float(scores[i])

#                 nxt = {p: next_use(trace, p, t + 1) for p in pages}
#                 correct = nxt[victim] == max(nxt.values())

#                 log.append({
#                     "t": t,
#                     "phase": "before" if t < shift else "after",
#                     "victim": victim,
#                     "confidence": confidence,
#                     "correct": correct,
#                     "text": explain(victim, rows[i][0], rows[i][1], confidence),
#                 })
#                 frames.remove(victim)
#             frames.add(page)
#         last_used[page] = t
        
#     return log

def learned_with_log(trace, num_frames, model, shift):
    frames = set()
    last_used = {}
    log = []
    for t, page in enumerate(trace):
        if page not in frames:
            if len(frames) == num_frames:
                pages = list(frames)
                rows = [features(trace, p, t, last_used) for p in pages]
                scores = model.predict_proba(rows)[:, 1]
                i = int(scores.argmax())
                victim = pages[i]
                confidence = float(scores[i])

                nxt = {p: next_use(trace, p, t + 1) for p in pages}
                correct = nxt[victim] == max(nxt.values())

                log.append({
                    "t": t,
                    "phase": "before" if t < shift else "after",
                    "victim": victim,
                    "confidence": confidence,
                    "correct": correct,
                    "ties": int((scores == scores.max()).sum()),
                    "text": explain(victim, rows[i][0], rows[i][1], confidence),
                })
                frames.remove(victim)
            frames.add(page)
        last_used[page] = t
    return log

def mean_or_nan(values):
    return statistics.mean(values) if values else float("nan")


def summarize(label, entries):
    right = [e["confidence"] for e in entries if e["correct"]]
    wrong = [e["confidence"] for e in entries if not e["correct"]]
    accuracy = len(right) / len(entries) if entries else float("nan")
    print(f"{label:8} decisions={len(entries):5}  accuracy={accuracy:.3f}  "
          f"conf when correct={mean_or_nan(right):.3f}  "
          f"conf when wrong={mean_or_nan(wrong):.3f}")


def calibration_rows(phase, entries):
    out = []
    for lo, hi in BINS:
        group = [e for e in entries if lo <= e["confidence"] < hi]
        if group:
            out.append({
                "phase": phase,
                "bin": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                "count": len(group),
                "mean_confidence": mean_or_nan([e["confidence"] for e in group]),
                "accuracy": sum(e["correct"] for e in group) / len(group),
            })
    return out


def main():
    os.makedirs("results", exist_ok=True)

    train_trace = locality_trace(1000, 50, rng=random.Random(TRAIN_SEED))
    model = train_model(train_trace, NUM_FRAMES)

    entries = []
    for seed in TEST_SEEDS:
        trace, shift = make_shift_trace(seed=seed)
        entries += learned_with_log(trace, NUM_FRAMES, model, shift)
        print("finished seed", seed)

    before = [e for e in entries if e["phase"] == "before"]
    after = [e for e in entries if e["phase"] == "after"]
    print("avg tied pages at top score  before:",
          round(statistics.mean(e["ties"] for e in before), 2),
          " after:",
          round(statistics.mean(e["ties"] for e in after), 2))
    print()
    summarize("overall", entries)
    summarize("before", before)
    summarize("after", after)
    print(f"chance accuracy for a random pick is about {1 / NUM_FRAMES:.2f}")

    rows = calibration_rows("before", before) + calibration_rows("after", after)
    print()
    for r in rows:
        print(f"{r['phase']:6} bin {r['bin']}: n={r['count']:5}  "
              f"mean conf={r['mean_confidence']:.3f}  accuracy={r['accuracy']:.3f}")

    with open("results/calibration.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open("results/explanations.txt", "w") as f:
        for e in before[:10] + after[:10]:
            outcome = "CORRECT" if e["correct"] else "WRONG"
            f.write(f"[t={e['t']}, {e['phase']} shift] {e['text']} -> {outcome}\n")

    print("\nsample explanations:")
    for e in (before[:2] + after[:2]):
        outcome = "CORRECT" if e["correct"] else "WRONG"
        print(f"[t={e['t']}, {e['phase']}] {e['text']} -> {outcome}")

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    for phase in ("before", "after"):
        pts = [r for r in rows if r["phase"] == phase]
        plt.plot([r["mean_confidence"] for r in pts],
                 [r["accuracy"] for r in pts],
                 marker="o", label=f"{phase} shift")
    plt.xlabel("mean confidence")
    plt.ylabel("accuracy (matches Optimal)")
    plt.title("Confidence vs correctness")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/calibration.png", dpi=150)
    print("\nsaved results/calibration.csv, explanations.txt, calibration.png")


if __name__ == "__main__":
    main()