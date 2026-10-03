
from collections import deque
from sklearn.tree import DecisionTreeClassifier

def fifo(trace, num_frames):
    frames = set()
    queue = deque()
    hits = []
    for page in trace:
        if page in frames:
            hits.append(True)
        else:
            hits.append(False)
            if len(frames) == num_frames:
                victim = queue.popleft()
                frames.remove(victim)
            frames.add(page)
            queue.append(page)
    return hits


def lru(trace, num_frames):
    last_used = {}
    hits = []
    for t, page in enumerate(trace):
        if page in last_used:
            hits.append(True)
        else:
            hits.append(False)
            if len(last_used) == num_frames:
                victim = min(last_used, key=last_used.get)
                del last_used[victim]
        last_used[page] = t
    return hits


INF = float("inf")


def next_use(trace, page, start):
    for i in range(start, len(trace)):
        if trace[i] == page:
            return i
    return INF


def optimal(trace, num_frames):
    frames = set()
    hits = []
    for t, page in enumerate(trace):
        if page in frames:
            hits.append(True)
        else:
            hits.append(False)
            if len(frames) == num_frames:
                victim = max(frames, key=lambda p: next_use(trace, p, t + 1))
                frames.remove(victim)
            frames.add(page)
    return hits



WINDOW = 50


def features(trace, page, t, last_used):
    recency = t - last_used[page]
    frequency = trace[max(0, t - WINDOW):t].count(page)
    return [recency, frequency]


def build_training_data(trace, num_frames):
    X, y = [], []
    frames = set()
    last_used = {}
    for t, page in enumerate(trace):
        if page not in frames:
            if len(frames) == num_frames:
                nxt = {p: next_use(trace, p, t + 1) for p in frames}
                farthest = max(nxt.values())
                for p in frames:
                    X.append(features(trace, p, t, last_used))
                    y.append(1 if nxt[p] == farthest else 0)
                victim = max(frames, key=lambda p: nxt[p])
                frames.remove(victim)
            frames.add(page)
        last_used[page] = t
    return X, y


def train_model(trace, num_frames, max_depth=4):
    X, y = build_training_data(trace, num_frames)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    model.fit(X, y)
    return model


def learned(trace, num_frames, model):
    frames = set()
    last_used = {}
    hits = []
    for t, page in enumerate(trace):
        if page in frames:
            hits.append(True)
        else:
            hits.append(False)
            if len(frames) == num_frames:
                pages = list(frames)
                X = [features(trace, p, t, last_used) for p in pages]
                scores = model.predict_proba(X)[:, 1]
                victim = pages[int(scores.argmax())]
                frames.remove(victim)
            frames.add(page)
        last_used[page] = t
    return hits




def summarize(name, hits, shift):
    for label, part in (("before", hits[:shift]), ("after", hits[shift:])):
        h = sum(part)
        print(f"{name:5} {label:6} hits={h:4} faults={len(part) - h:4} "
              f"hit ratio={h / len(part):.3f}")


if __name__ == "__main__":
    demo = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    print("FIFO faults   :", fifo(demo, 3).count(False))      # expect 9
    print("LRU faults    :", lru(demo, 3).count(False))       # expect 10
    print("Optimal faults:", optimal(demo, 3).count(False))   # expect 7

    from traces import make_shift_trace
    trace, shift = make_shift_trace()
    summarize("FIFO", fifo(trace, 10), shift)
    summarize("LRU", lru(trace, 10), shift)
    summarize("OPT", optimal(trace, 10), shift)