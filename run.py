import csv
import os
import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from traces import make_shift_trace, locality_trace
from policies import fifo, lru, optimal, train_model, learned

NUM_FRAMES = 10
TRAIN_SEED = 1
TEST_SEEDS = range(100, 110)


def split_stats(hits, shift):
    before, after = hits[:shift], hits[shift:]
    return {
        "before_ratio": sum(before) / len(before),
        "after_ratio": sum(after) / len(after),
        "before_faults": len(before) - sum(before),
        "after_faults": len(after) - sum(after),
    }


def avg(rows, key):
    values = [r[key] for r in rows]
    return statistics.mean(values), statistics.stdev(values)


def main():
    os.makedirs("results", exist_ok=True)

    train_trace = locality_trace(1000, 50, rng=random.Random(TRAIN_SEED))
    model = train_model(train_trace, NUM_FRAMES)

    results = {"FIFO": [], "LRU": [], "OPT": [], "LEARN": []}
    for seed in TEST_SEEDS:
        trace, shift = make_shift_trace(seed=seed)
        runs = {
            "FIFO": fifo(trace, NUM_FRAMES),
            "LRU": lru(trace, NUM_FRAMES),
            "OPT": optimal(trace, NUM_FRAMES),
            "LEARN": learned(trace, NUM_FRAMES, model),
        }
        for name, hits in runs.items():
            results[name].append(split_stats(hits, shift))
        print("finished seed", seed)

    summary = []
    for name, rows in results.items():
        b_mean, b_std = avg(rows, "before_ratio")
        a_mean, a_std = avg(rows, "after_ratio")
        b_faults, _ = avg(rows, "before_faults")
        a_faults, _ = avg(rows, "after_faults")
        summary.append({
            "policy": name,
            "before_ratio": b_mean, "before_std": b_std,
            "after_ratio": a_mean, "after_std": a_std,
            "drop": b_mean - a_mean,
            "before_faults": b_faults, "after_faults": a_faults,
        })

    print()
    print(f"{'policy':6} {'before':>8} {'after':>8} {'drop':>8} "
          f"{'faults before':>14} {'faults after':>13}")
    for s in summary:
        print(f"{s['policy']:6} {s['before_ratio']:8.3f} {s['after_ratio']:8.3f} "
              f"{s['drop']:8.3f} {s['before_faults']:14.1f} {s['after_faults']:13.1f}")

    with open("results/summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    names = [s["policy"] for s in summary]
    x = list(range(len(names)))
    w = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar([i - w / 2 for i in x], [s["before_ratio"] for s in summary], w,
            yerr=[s["before_std"] for s in summary], capsize=3, label="before shift")
    plt.bar([i + w / 2 for i in x], [s["after_ratio"] for s in summary], w,
            yerr=[s["after_std"] for s in summary], capsize=3, label="after shift")
    plt.xticks(x, names)
    plt.ylabel("hit ratio")
    plt.title("Hit ratio before vs after workload shift (mean of 10 traces)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/hit_ratio.png", dpi=150)
    print("\nsaved results/summary.csv and results/hit_ratio.png")


if __name__ == "__main__":
    main()