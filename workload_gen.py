"""Synthetic row-id traces for the product-key memory buffer pool.

Usage:
    python workload_gen.py gen   --out traces/z1 --alpha 1.0 --drift 0.01
    python workload_gen.py gen   --out traces/pkm --mode pkm --rows 16777216 --alpha 0.8
    python workload_gen.py stats traces/z1
    python workload_gen.py test

Knobs:
    alpha     skew. Bounded Zipf, P(rank r) ∝ (r+1)^-alpha. 0 = uniform; ~1 = heavy skew.
    drift     reuse across batches. After each batch, this fraction of the hot set
              (top hot_frac of ranks) is swapped with random cold rows. 0 = hot set never
              changes (best case for a cache); larger = working set moves faster.
    mode      iid: every id is an independent Zipf draw over all rows.
              pkm: product-key structure. Each (token, head) draws k distinct subkeys per
                   half from Zipf over n = sqrt(rows) subkeys (Gumbel-top-k, so draws are
                   distinct like real top-k), scores each by log p + Gumbel noise, and keeps
                   the best k of the k^2 pairs; id = i*n + j. Each head has its own subkey
                   permutation, as each head has its own C1, C2. Skew compounds across the
                   two halves, so the row-level skew is higher than alpha suggests.
Hot rows are spread over the table by a random rank→id permutation (no spatial locality).

Output (in --out):
    trace.bin   uint32 little-endian, shape [batches, T, h, k] (row-major)
    meta.json   all parameters + seed
`pytorch_ref.py replay --trace OUT/trace.bin --k K` produces golden outputs for a trace.
"""
import argparse
import json
import math
import os
import tempfile
import time
from collections import OrderedDict

import numpy as np

ROW_BYTES = 1024  # 512 fp16 values


# --------------------------------------------------------------------------- sampling

class ZipfRanks:
    """Bounded Zipf over ranks 0..n-1: P(r) ∝ (r+1)^-alpha."""

    def __init__(self, n, alpha):
        w = np.arange(1, n + 1, dtype=np.float64) ** -alpha
        w /= w.sum()
        self.n = n
        self.p = w
        self.logp = np.log(w)
        self.cdf = np.cumsum(w)
        self.cdf[-1] = 1.0

    def sample(self, rng, m):
        """m ranks, with replacement."""
        return np.minimum(np.searchsorted(self.cdf, rng.random(m), side="right"), self.n - 1)

    def sample_distinct(self, rng, rows, k, chunk=512):
        """rows x k ranks, distinct within each row (sampling without replacement ∝ p).

        Gumbel-top-k: the k largest log p + Gumbel are an exact draw without replacement.
        The perturbed values are returned as scores, used to pick pairs in pkm mode.
        """
        ranks = np.empty((rows, k), np.int64)
        scores = np.empty((rows, k), np.float32)
        logp = self.logp.astype(np.float32)
        for s in range(0, rows, chunk):
            e = min(rows, s + chunk)
            g = logp + rng.gumbel(size=(e - s, self.n)).astype(np.float32)
            top = np.argpartition(-g, k - 1, axis=1)[:, :k]
            ranks[s:e] = top
            scores[s:e] = np.take_along_axis(g, top, 1)
        return ranks, scores


def drift_perm(perm, rng, hot, drift):
    """Swap round(drift*hot) random hot ranks with random cold ranks, in place.

    Uses stochastic rounding so small drift values still move rows on average.
    """
    x = drift * hot
    m = int(x) + (rng.random() < x - int(x))
    m = min(m, hot, len(perm) - hot)
    if m <= 0:
        return
    a = rng.choice(hot, m, replace=False)
    b = hot + rng.choice(len(perm) - hot, m, replace=False)
    perm[a], perm[b] = perm[b], perm[a].copy()


# --------------------------------------------------------------------------- generator

def generate(out, mode="iid", rows=1 << 22, batches=100, T=1024, h=4, k=32,
             alpha=1.0, drift=0.0, hot_frac=0.01, seed=0, quiet=False):
    rng = np.random.default_rng(seed)
    per_batch = T * h * k
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    if mode == "iid":
        if rows > 1 << 32:
            raise ValueError("rows must fit in uint32")
        zipf = ZipfRanks(rows, alpha)
        perm = rng.permutation(rows).astype(np.uint32)
        hot = max(1, int(hot_frac * rows))

        def next_batch():
            ids = perm[zipf.sample(rng, per_batch)]
            drift_perm(perm, rng, hot, drift)
            return ids
    elif mode == "pkm":
        n = math.isqrt(rows)
        if n * n != rows:
            raise ValueError(f"pkm mode needs rows = n^2, got {rows}")
        if k > n:
            raise ValueError(f"k={k} > n={n}")
        zipf = ZipfRanks(n, alpha)
        perm1 = np.stack([rng.permutation(n) for _ in range(h)])  # [h, n] rank → subkey
        perm2 = np.stack([rng.permutation(n) for _ in range(h)])
        hot = max(1, int(hot_frac * n))
        M = T * h
        head = np.arange(M) % h  # layout [T, h]: flattened row t*h + head

        def next_batch():
            r1, s1 = zipf.sample_distinct(rng, M, k)
            r2, s2 = zipf.sample_distinct(rng, M, k)
            sub1 = perm1[head[:, None], r1]
            sub2 = perm2[head[:, None], r2]
            cand = (s1[:, :, None] + s2[:, None, :]).reshape(M, k * k)
            top = np.argpartition(-cand, k - 1, axis=1)[:, :k]
            i = np.take_along_axis(sub1, top // k, 1)
            j = np.take_along_axis(sub2, top % k, 1)
            for hh in range(h):
                drift_perm(perm1[hh], rng, hot, drift)
                drift_perm(perm2[hh], rng, hot, drift)
            return (i * n + j).astype(np.uint32).ravel()
    else:
        raise ValueError(f"unknown mode {mode}")

    with open(os.path.join(out, "trace.bin"), "wb") as f:
        for b in range(batches):
            next_batch().astype("<u4").tofile(f)
            if not quiet and (b + 1) % max(1, batches // 10) == 0:
                print(f"  batch {b + 1}/{batches}  ({time.time() - t0:.1f}s)")

    meta = dict(mode=mode, rows=rows, batches=batches, T=T, h=h, k=k, alpha=alpha, drift=drift,
                hot_frac=hot_frac, seed=seed, ids_per_batch=per_batch, row_bytes=ROW_BYTES,
                layout="[batches, T, h, k] uint32 little-endian")
    if mode == "pkm":
        meta["n"] = math.isqrt(rows)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if not quiet:
        size = batches * per_batch * 4 / 2**20
        print(f"wrote {out}/trace.bin: {batches} batches × {per_batch} ids ({size:.1f} MB) "
              f"in {time.time() - t0:.1f}s")
    return meta


def load_trace(out):
    with open(os.path.join(out, "meta.json")) as f:
        meta = json.load(f)
    trace = np.memmap(os.path.join(out, "trace.bin"), dtype="<u4", mode="r")
    return trace.reshape(meta["batches"], meta["ids_per_batch"]), meta


# --------------------------------------------------------------------------- stats

def lru_distances(trace, rows):
    """LRU stack distance of every (batch, unique id) access, at batch granularity.

    Models the buffer pool: a batch is deduplicated (sorted unique), every unique id is
    probed against the cache state left by the previous batch, then the batch's ids are
    touched in sorted order. An id hits in an LRU cache of C rows iff fewer than C distinct
    ids were touched since its last access, i.e. distance < C. First touch → distance = inf.

    Yields (unique_ids, counts, distances) per batch.
    """
    last = np.full(rows, -1, np.int64)            # id → timestamp of last touch
    alive = np.zeros(trace.size, np.int8)          # timestamp → still someone's last touch
    now = 0
    for batch in trace:
        u, cnt = np.unique(batch, return_counts=True)
        t = last[u]
        seen = t >= 0
        dist = np.full(len(u), np.inf)
        if now and seen.any():
            cs = np.cumsum(alive[:now], dtype=np.int64)
            dist[seen] = cs[now - 1] - cs[t[seen]]  # alive timestamps in (t, now)
        alive[t[seen]] = 0
        ts = now + np.arange(len(u))
        alive[ts] = 1
        last[u] = ts
        now += len(u)
        yield u, cnt, dist


def compute_stats(out, cache_fracs=(0.01, 0.05, 0.10, 0.25, 0.50), warmup=1):
    trace, meta = load_trace(out)
    rows, B = meta["rows"], meta["batches"]
    warmup = min(warmup, B - 1)
    sizes = [max(1, int(f * rows)) for f in cache_fracs]
    row_bytes = meta.get("row_bytes", ROW_BYTES)

    uniq_per_batch = []
    freq = np.zeros(rows, np.int64)       # batches containing each id (after warmup)
    lru_uh = np.zeros(len(sizes))        # unique hits
    lru_rh = np.zeros(len(sizes))        # read hits (weighted by duplicates in batch)
    tot_u = tot_r = 0
    for b, (u, cnt, dist) in enumerate(lru_distances(trace, rows)):
        uniq_per_batch.append(len(u))
        if b < warmup:
            continue
        freq[u] += 1
        tot_u += len(u)
        tot_r += int(cnt.sum())
        for i, C in enumerate(sizes):
            hit = dist < C
            lru_uh[i] += hit.sum()
            lru_rh[i] += cnt[hit].sum()

    # Static oracle: cache permanently holds the C ids present in the most batches.
    static_uh = []
    order = np.argsort(-freq, kind="stable")
    cum = np.cumsum(freq[order])
    for C in sizes:
        static_uh.append(cum[min(C, rows) - 1] / tot_u)

    upb = np.array(uniq_per_batch)
    res = dict(
        trace=os.path.abspath(out), meta=meta, warmup_batches=warmup,
        ids_per_batch=meta["ids_per_batch"],
        unique_per_batch_mean=float(upb.mean()), unique_per_batch_min=int(upb.min()),
        unique_per_batch_max=int(upb.max()),
        unique_fraction=float(upb.mean() / meta["ids_per_batch"]),
        distinct_ids_in_trace=int((freq > 0).sum()),
        caches=[dict(frac=f, rows=C,
                     lru_unique_hit=float(lru_uh[i] / tot_u),
                     lru_read_hit=float(lru_rh[i] / tot_r),
                     static_opt_unique_hit=float(static_uh[i]),
                     lru_miss_mb_per_batch=float((tot_u - lru_uh[i]) * row_bytes / (B - warmup) / 2**20))
                for i, (f, C) in enumerate(zip(cache_fracs, sizes))],
    )
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def print_stats(res):
    m = res["meta"]
    print(f"{res['trace']}: mode={m['mode']} rows={m['rows']} alpha={m['alpha']} drift={m['drift']} "
          f"batches={m['batches']} ids/batch={res['ids_per_batch']}")
    print(f"unique/batch: mean {res['unique_per_batch_mean']:.0f} "
          f"({100 * res['unique_fraction']:.1f}% of ids, dedup saves "
          f"{1 / res['unique_fraction']:.2f}x), min {res['unique_per_batch_min']}, "
          f"max {res['unique_per_batch_max']}")
    print(f"distinct ids in trace (after {res['warmup_batches']} warmup batches): "
          f"{res['distinct_ids_in_trace']} ({100 * res['distinct_ids_in_trace'] / m['rows']:.1f}% of table)")
    print(f"{'cache':>7} {'rows':>10} {'LRU uniq-hit':>13} {'LRU read-hit':>13} "
          f"{'static-opt':>11} {'LRU miss MB/batch':>18}")
    for c in res["caches"]:
        print(f"{100 * c['frac']:6.1f}% {c['rows']:>10} {c['lru_unique_hit']:>13.3f} "
              f"{c['lru_read_hit']:>13.3f} {c['static_opt_unique_hit']:>11.3f} "
              f"{c['lru_miss_mb_per_batch']:>18.2f}")


# --------------------------------------------------------------------------- self-tests

def lru_reference(trace, C):
    """Plain OrderedDict LRU with the same batch semantics as lru_distances."""
    cache, hits = OrderedDict(), []
    for batch in trace:
        u = np.unique(batch)
        hits.append(np.array([x in cache for x in u.tolist()]))
        for x in u.tolist():
            cache[x] = True
            cache.move_to_end(x)
            if len(cache) > C:
                cache.popitem(last=False)
    return hits


def run_tests():
    rng = np.random.default_rng(0)

    # 1. Zipf sampler matches its distribution
    z = ZipfRanks(1000, 1.0)
    s = z.sample(rng, 1_000_000)
    emp = np.bincount(s, minlength=1000) / s.size
    assert abs(emp[0] - z.p[0]) / z.p[0] < 0.02 and abs(emp[9] - z.p[9]) / z.p[9] < 0.05
    assert s.min() >= 0 and s.max() < 1000
    u = ZipfRanks(1000, 0.0).sample(rng, 1_000_000)
    assert np.bincount(u, minlength=1000).std() / 1000 < 0.1, "alpha=0 should be uniform"
    print("ok  Zipf sampler matches P(r) ∝ (r+1)^-alpha; alpha=0 is uniform")

    # 2. distinct sampling: distinct per row, in range, and still skewed
    r, _ = z.sample_distinct(rng, 2000, 16)
    assert all(len(set(row)) == 16 for row in r.tolist()) and r.max() < 1000
    assert (r == 0).any(axis=1).mean() > 0.5, "rank 0 should appear in most rows"
    print("ok  Gumbel-top-k draws are distinct per row and skewed")

    # 3. drift
    p = np.arange(100, dtype=np.uint32)
    q = p.copy(); drift_perm(q, rng, 10, 0.0)
    assert np.array_equal(p, q), "drift=0 must not change the permutation"
    q = p.copy(); drift_perm(q, rng, 10, 1.0)
    assert np.isin(q[:10], p[:10]).sum() == 0 and np.array_equal(np.sort(q), p)
    q = p.copy(); drift_perm(q, rng, 10, 0.3)
    assert (q[:10] != p[:10]).sum() == 3 and np.array_equal(np.sort(q), p)
    print("ok  drift swaps the right number of hot ranks and keeps a permutation")

    with tempfile.TemporaryDirectory() as tmp:
        # 4. iid trace: shape, range, file layout
        d = os.path.join(tmp, "iid")
        meta = generate(d, "iid", rows=4096, batches=5, T=16, h=2, k=8, alpha=1.0, seed=1, quiet=True)
        tr, _ = load_trace(d)
        assert tr.shape == (5, 16 * 2 * 8) and tr.max() < 4096
        assert os.path.getsize(os.path.join(d, "trace.bin")) == 5 * meta["ids_per_batch"] * 4
        # same seed → same trace
        generate(os.path.join(tmp, "iid2"), "iid", rows=4096, batches=5, T=16, h=2, k=8,
                 alpha=1.0, seed=1, quiet=True)
        assert np.array_equal(tr, load_trace(os.path.join(tmp, "iid2"))[0])
        print("ok  iid trace shape, range, size on disk, reproducible from seed")

        # 5. pkm trace: distinct ids per (token, head), in range
        d = os.path.join(tmp, "pkm")
        generate(d, "pkm", rows=64 * 64, batches=3, T=16, h=2, k=8, alpha=0.8, drift=0.1,
                 seed=2, quiet=True)
        tr, _ = load_trace(d)
        per = tr.reshape(3, 16, 2, 8)
        assert tr.max() < 4096
        assert all(len(set(x)) == 8 for x in per.reshape(-1, 8).tolist())
        print("ok  pkm trace: ids in range, distinct within each (token, head)")

        # 6. vectorized LRU == OrderedDict LRU
        d = os.path.join(tmp, "lru")
        generate(d, "iid", rows=2000, batches=30, T=8, h=2, k=8, alpha=0.9, drift=0.2,
                 hot_frac=0.05, seed=3, quiet=True)
        tr, _ = load_trace(d)
        dists = [dist for _, _, dist in lru_distances(tr, 2000)]
        for C in (1, 10, 50, 200, 2000):
            ref = lru_reference(tr, C)
            for b in range(len(tr)):
                assert np.array_equal(dists[b] < C, ref[b]), f"LRU mismatch C={C} batch={b}"
        res = compute_stats(d, cache_fracs=(0.01, 0.1, 1.0), warmup=1)
        hits = [c["lru_unique_hit"] for c in res["caches"]]
        assert hits == sorted(hits), "hit rate must grow with cache size"
        assert all(c["static_opt_unique_hit"] <= 1 for c in res["caches"])
        print("ok  vectorized LRU stack distances match an OrderedDict LRU")

        # 7. skew and drift move hit rate the way they should
        def hit(alpha, drift):
            d = os.path.join(tmp, f"a{alpha}d{drift}")
            generate(d, "iid", rows=20000, batches=20, T=32, h=2, k=8, alpha=alpha, drift=drift,
                     hot_frac=0.05, seed=4, quiet=True)
            return compute_stats(d, cache_fracs=(0.05,))["caches"][0]["lru_unique_hit"]
        assert hit(0.0, 0.0) < hit(0.8, 0.0) < hit(1.2, 0.0)
        assert hit(1.0, 0.5) < hit(1.0, 0.0)
        print("ok  more skew → higher hit rate; more drift → lower hit rate")
    print("all tests passed")


# --------------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a trace")
    g.add_argument("--out", required=True)
    g.add_argument("--mode", choices=("iid", "pkm"), default="iid")
    g.add_argument("--rows", type=int, default=1 << 22, help="table rows (pkm: must be n^2)")
    g.add_argument("--batches", type=int, default=100)
    g.add_argument("--T", type=int, default=1024, help="tokens per batch")
    g.add_argument("--h", type=int, default=4, help="heads")
    g.add_argument("--k", type=int, default=32, help="rows selected per token per head")
    g.add_argument("--alpha", type=float, default=1.0, help="Zipf skew (0 = uniform)")
    g.add_argument("--drift", type=float, default=0.0, help="fraction of hot set replaced per batch")
    g.add_argument("--hot-frac", type=float, default=0.01, help="hot set size as a fraction of ranks")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--stats", action="store_true", help="also compute and print stats")

    s = sub.add_parser("stats", help="unique fraction and LRU / static-oracle hit rates")
    s.add_argument("dir")
    s.add_argument("--cache-fracs", type=lambda x: tuple(float(v) for v in x.split(",")),
                   default=(0.01, 0.05, 0.10, 0.25, 0.50))
    s.add_argument("--warmup", type=int, default=1, help="batches excluded from hit rates")

    sub.add_parser("test", help="run self-checks")

    a = p.parse_args()
    if a.cmd == "gen":
        generate(a.out, a.mode, a.rows, a.batches, a.T, a.h, a.k, a.alpha, a.drift, a.hot_frac, a.seed)
        if a.stats:
            print_stats(compute_stats(a.out))
    elif a.cmd == "stats":
        print_stats(compute_stats(a.dir, a.cache_fracs, a.warmup))
    else:
        run_tests()


if __name__ == "__main__":
    main()
