import random


def locality_trace(length, num_pages, ws_size=5, stay_prob=0.9, rng=None):
    rng = rng or random.Random()
    start = rng.randrange(num_pages)
    working_set = [(start + i) % num_pages
                   for i in range(ws_size)]
    trace = []
    for _ in range(length):
        if rng.random() < stay_prob:
            page = rng.choice(working_set)
        else:
            page = rng.randrange(num_pages)
        trace.append(page)
    return trace


def random_trace(length, num_pages, rng=None):
    rng = rng or random.Random()
    return [rng.randrange(num_pages) for _ in range(length)]


def make_shift_trace(length=2000, num_pages=50, seed=42):
    rng = random.Random(seed)
    half = length // 2
    first = locality_trace(half, num_pages, rng=rng)
    second = random_trace(length - half, num_pages, rng=rng)
    return first + second, half


if __name__ == "__main__":
    from collections import Counter
    trace, shift = make_shift_trace()
    print("shift at index:", shift)
    print("start :", trace[:25])
    print("end   :", trace[-25:])
    top5 = sum(c for _, c in Counter(trace[:shift]).most_common(5))
    print("first half: share of requests on top 5 pages =", top5 / shift)
    top5 = sum(c for _, c in Counter(trace[shift:]).most_common(5))
    print("second half: share of requests on top 5 pages =", top5 / (len(trace) - shift))