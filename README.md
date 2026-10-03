## Setup

Requires Python 3.8 or newer.

```
git clone https://github.com/sumaiyafahmida2610-sudo/os-term-paper.git
cd os-term-paper
pip install scikit-learn matplotlib
```

## How to run

Run these from the repository folder:

```
python traces.py     # sanity check of the trace generator
python policies.py   # demo example, single-trace comparison, learned tree rules
python run.py        # 10-trace experiment -> results/summary.csv, results/hit_ratio.png
python explain.py    # bonus analysis -> results/calibration.csv, calibration.png, explanations.txt
```

To repeat the memory-size experiment, change `NUM_FRAMES` at the top of `run.py`
(I used 5, 10, and 20), save, and run `python run.py` again. Each run overwrites
`results/summary.csv`, so copy it first if you want to keep it
(for example `cp results/summary.csv results/summary_frames5.csv`).

All random traces use fixed seeds (training seed 1, test seeds 100-109), so
results are reproducible.

## AI assistance disclosure

I used Claude (Anthropic) as a learning and coding assistant for this project.

Claude helped with:
- Explaining how FIFO, LRU, and Optimal page replacement work
- Suggesting Track 1 and the overall design: the locality-then-random trace,
  recency and frequency as features, labels taken from Optimal's eviction
  choices, a decision tree, and training and testing on different seeds
- Writing the code in this repository (`traces.py`, `policies.py`, `run.py`,
  `explain.py`), which I went through step by step with its explanations
- Suggesting the memory-size experiment (5, 10, and 20 frames) and drafting a
  plain-text summary of its results, which I reviewed
- The template-based explanations and the confidence-versus-correctness
  analysis for the bonus part

I did myself:
- Created and organized the repository
- Ran all experiments, including the additional memory-size experiment, and
  collected the results in `results/`
-  I did a expectation vs observation analysis on three test. Here is what I noted:
## Expectation vs. Observed Results

Before running the memory-size experiment, we expected the following:

* After the workload shift, FIFO, LRU, and LEARN were expected to achieve a hit ratio close to `frames / 50`, because the second half of the workload is uniformly random over 50 pages.
* OPT was expected to remain above `frames / 50` because it has knowledge of future requests.
* Before the shift, the 5-page working set was expected to give high hit ratios, but performance was expected to be noticeably worse with only 5 frames because the 10% random requests could evict useful working-set pages.
* With 20 frames, all policies were expected to perform well and become closer to each other because the working set would fit comfortably in memory.
* The learned policy was expected to behave similarly to LRU, although it might perform differently when memory was very tight.
* The advantage of choosing a particular replacement policy was expected to become smaller as the number of frames increased.

### What the Results Showed

The results strongly confirmed the main expectations.

**1. After the shift, FIFO, LRU, and LEARN matched `frames / 50`.**

| Frames | Expected | FIFO  | LRU   | LEARN |
| ------ | -------- | ----- | ----- | ----- |
| 5      | 0.10     | 0.096 | 0.095 | 0.093 |
| 10     | 0.20     | 0.198 | 0.197 | 0.197 |
| 20     | 0.40     | 0.396 | 0.399 | 0.400 |

The observed values are within about 0.005 of the expected values. This confirms that, once the workload becomes uniformly random, policies that only use past information cannot significantly improve the hit ratio through a smarter replacement strategy. Increasing the number of frames is the main way to improve performance.

**2. OPT performed much better than expected history-based policies after the shift.**

OPT achieved hit ratios of:

* 0.314 with 5 frames
* 0.490 with 10 frames
* 0.704 with 20 frames

This confirms the expected advantage of knowing future requests. Unlike FIFO, LRU, and LEARN, OPT can use future information to decide which page to evict.

**3. Before the shift, 5 frames produced noticeably worse performance.**

The observed hit ratios were:

* FIFO: 0.701
* LRU: 0.761
* LEARN: 0.837
* OPT: 0.861

The initial estimate for FIFO and LRU was around 0.75–0.85. LRU was close to this range, while FIFO performed worse at 0.701. The learned policy performed considerably better than initially expected.

**4. With 20 frames, the policies became much closer.**

The results were:

* FIFO: 0.911
* LRU: 0.929
* LEARN: 0.928
* OPT: 0.948

This matches the expectation that extra memory reduces the importance of the replacement policy. The 5-page working set fits easily, so FIFO, LRU, LEARN, and OPT have much smaller differences.

**5. LEARN showed an important difference from LRU at 5 frames.**

At 5 frames:

* LRU = 0.761
* LEARN = 0.837

The learned policy was substantially better than LRU.

However, at 10 and 20 frames, they were almost identical:

* 10 frames: LRU = 0.910, LEARN = 0.909
* 20 frames: LRU = 0.929, LEARN = 0.928

This suggests that the learned policy provides its main benefit when memory is tight. A possible explanation is that the learned policy uses access frequency in addition to recency, allowing it to distinguish one-off random pages from frequently accessed working-set pages. With more memory, the extra frames reduce the need for this distinction.

This explanation is a hypothesis and would need further experiments to confirm.

**6. The expected reduction in policy differences with more memory was confirmed.**

The gap between OPT and FIFO before the shift decreased as memory increased:

* 5 frames: 0.861 - 0.701 = 0.160
* 10 frames: 0.931 - 0.852 = 0.079
* 20 frames: 0.948 - 0.911 = 0.037

Therefore, replacement-policy choice matters most when memory is limited.

### Overall Conclusion

The experiment largely confirmed the original expectations.

The strongest result is that after the workload changes to uniform random access, FIFO, LRU, and LEARN all converge to approximately `frames / 50`, regardless of how sophisticated the replacement policy is. More memory is therefore the main remedy for the random-access phase.

Before the shift, however, the replacement policy still matters, especially when memory is tight. The learned policy achieved a clear advantage over LRU with only 5 frames, while this advantage disappeared when more memory was available.

OPT remained the best-performing policy in every case because it has access to future information. This demonstrates the difference between improving replacement decisions using historical information and having actual knowledge of future requests.

One result that deserves further investigation is why the learned policy significantly outperformed LRU at 5 frames but not at 10 or 20 frames. Inspecting the learned decision tree and testing whether frequency is used near the top of the tree could help explain this behavior.
## Bonus: explanation confidence (completed)

I implemented the optional bonus track in `explain.py`. For every eviction made
by the learned policy, it generates a natural-language explanation using a
rule-based template (no LLM API was used), along with a confidence score (the
decision tree's score for the evicted page). It then compares confidence against
whether the decision was correct (matched Optimal's choice).

Outputs: `results/explanations.txt`, `results/calibration.csv`,
`results/calibration.png`. Finding: confidence was nearly identical for correct
and wrong decisions, so it was uninformative and overconfident.