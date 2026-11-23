"""PyTorch reference for product-key memory lookup.

Usage:
    python pytorch_ref.py test                                   # self-checks (CPU is fine)
    python pytorch_ref.py dump  --preset small --out golden/small
    python pytorch_ref.py replay --trace trace.bin --k 32 --out golden/trace
    python pytorch_ref.py bench --rows 4194304 --lookups 131072  # needs CUDA for meaningful numbers

Golden-file format (all raw little-endian, row-major, readable with fread):
    ids.bin  uint32  [N, k]       row ids into the value table
    w.bin    float32 [N, k]       softmax weights
    y.bin    float32 [N, width]   weighted sum of rows (fp32 accumulation)
    meta.json                     shapes, seed, preset, table kind
N = T*h for `dump`, or (trace length / k) for `replay`.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

WIDTH = 512  # values per row; fp16 → 1 KB rows

PRESETS = {
    # name:   T,    h, d,   n,    k
    "tiny":  (8,    2, 32,  64,   8),    # 4096 rows; brute force is cheap
    "small": (256,  4, 256, 256,  32),   # 65536 rows
    "paper": (1024, 4, 512, 4096, 32),   # 2^24 rows (16 GB table), Lample-style k/h
}


# --------------------------------------------------------------------------- value table

def synthetic_rows(ids, width=WIDTH):
    """Row values computed from the row id, so a 16 GB table never has to exist.

    Same formula in CUDA (uint32 wraparound gives identical low bits):
        __half v = __float2half(((r * 2654435761u + c * 40503u) & 1023u) / 1024.0f - 0.5f);
    Values are multiples of 1/1024 in [-0.5, 0.5), all exactly representable in fp16.
    """
    r = ids.long()[..., None]
    c = torch.arange(width, device=ids.device)
    return (((r * 2654435761 + c * 40503) & 1023).float() / 1024 - 0.5).half()


def materialize_table(n_rows, width=WIDTH, device="cpu"):
    """Full fp16 table with the same values as synthetic_rows (small tables only)."""
    return synthetic_rows(torch.arange(n_rows, device=device), width)


# --------------------------------------------------------------------------- lookup

def weighted_gather(ids, w, V=None, width=WIDTH, chunk_rows=1 << 16):
    """y[..., :] = sum_j w[..., j] * V[ids[..., j]]  — the part the CUDA kernels implement.

    V=None uses synthetic_rows. Work is chunked so memory stays bounded for large batches.
    """
    lead, k = ids.shape[:-1], ids.shape[-1]
    if V is not None:
        width = V.shape[-1]
    ids2 = ids.reshape(-1, k)
    w2 = w.reshape(-1, k).float()
    out = torch.empty(ids2.shape[0], width, dtype=torch.float32, device=ids.device)
    step = max(1, chunk_rows // k)
    for s in range(0, ids2.shape[0], step):
        idx = ids2[s:s + step]
        rows = V[idx] if V is not None else synthetic_rows(idx, width)
        out[s:s + step] = (w2[s:s + step, :, None] * rows.float()).sum(1)
    return out.reshape(*lead, width)


def pkm_select(q, C1, C2, k):
    """Product-key top-k. q:[T,h,d], C1,C2:[h,n,d/2] → ids:[T,h,k] (int64), w:[T,h,k] (fp32)."""
    n = C1.shape[1]
    q1, q2 = q.chunk(2, dim=-1)
    s1 = torch.einsum('thd,hnd->thn', q1, C1).float()
    s2 = torch.einsum('thd,hnd->thn', q2, C2).float()
    v1, i1 = s1.topk(k, -1)
    v2, i2 = s2.topk(k, -1)
    cand = (v1[..., :, None] + v2[..., None, :]).flatten(-2)  # [T,h,k*k]
    sc, flat = cand.topk(k, -1)
    ids = i1.gather(-1, flat // k) * n + i2.gather(-1, flat % k)
    w = sc.softmax(-1)
    return ids, w


def pkm_lookup(q, C1, C2, V, k):
    """Full layer read. V=None uses the synthetic table. Returns y:[T,h,width], ids, w."""
    ids, w = pkm_select(q, C1, C2, k)
    return weighted_gather(ids, w, V), ids, w


def make_inputs(T, h, d, n, seed=0, device="cpu", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(T, h, d, generator=g)
    C1 = torch.randn(h, n, d // 2, generator=g) / (d // 2) ** 0.5
    C2 = torch.randn(h, n, d // 2, generator=g) / (d // 2) ** 0.5
    return tuple(x.to(device=device, dtype=dtype) for x in (q, C1, C2))


# --------------------------------------------------------------------------- golden files

def write_golden(out_dir, ids, w, y, meta):
    os.makedirs(out_dir, exist_ok=True)
    k = ids.shape[-1]
    ids_np = ids.reshape(-1, k).cpu().numpy().astype('<u4')
    w_np = w.reshape(-1, k).float().cpu().numpy().astype('<f4')
    y_np = y.reshape(ids_np.shape[0], -1).float().cpu().numpy().astype('<f4')
    ids_np.tofile(os.path.join(out_dir, "ids.bin"))
    w_np.tofile(os.path.join(out_dir, "w.bin"))
    y_np.tofile(os.path.join(out_dir, "y.bin"))
    meta = dict(meta, N=int(ids_np.shape[0]), k=int(k), width=int(y_np.shape[1]),
                ids_dtype="uint32", w_dtype="float32", y_dtype="float32")
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_dir}: N={meta['N']} k={k} width={meta['width']}")


def load_golden(out_dir):
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    N, k, width = meta["N"], meta["k"], meta["width"]
    ids = np.fromfile(os.path.join(out_dir, "ids.bin"), dtype='<u4').reshape(N, k)
    w = np.fromfile(os.path.join(out_dir, "w.bin"), dtype='<f4').reshape(N, k)
    y = np.fromfile(os.path.join(out_dir, "y.bin"), dtype='<f4').reshape(N, width)
    return ids, w, y, meta


# --------------------------------------------------------------------------- self-tests

def run_tests(device="cpu"):
    T, h, d, n, k = PRESETS["tiny"]
    q, C1, C2 = make_inputs(T, h, d, n, seed=1, device=device)
    V = materialize_table(n * n, device=device)

    y, ids, w = pkm_lookup(q, C1, C2, V, k)
    y_syn, ids_syn, _ = pkm_lookup(q, C1, C2, None, k)

    # 1. shapes and ranges
    assert ids.shape == (T, h, k) and w.shape == (T, h, k) and y.shape == (T, h, WIDTH)
    assert ids.min() >= 0 and ids.max() < n * n, "row id out of range"
    assert torch.allclose(w.sum(-1), torch.ones(T, h, device=device), atol=1e-6), "weights must sum to 1"
    # ids within one (t, h) are distinct pairs
    assert (ids.sort(-1).values.diff(dim=-1) != 0).all(), "duplicate ids within a token/head"
    print("ok  shapes, id range, weights sum to 1, distinct ids per token/head")

    # 2. product-key shortcut == brute force over all n^2 keys
    q1, q2 = q.chunk(2, dim=-1)
    s1 = torch.einsum('thd,hnd->thn', q1, C1).float()
    s2 = torch.einsum('thd,hnd->thn', q2, C2).float()
    full = (s1[..., :, None] + s2[..., None, :]).flatten(-2)  # [T,h,n*n]; id = i*n + j
    ref_sc, ref_ids = full.topk(k, -1)
    assert torch.equal(ref_ids.sort(-1).values, ids.sort(-1).values), "product-key top-k != brute force"
    assert torch.allclose(ref_sc.softmax(-1), w, atol=1e-6)
    print("ok  product-key top-k matches brute-force top-k over n^2 keys")

    # 3. synthetic rows == materialized table (CUDA side can use either)
    assert torch.equal(ids, ids_syn)
    assert torch.equal(y, y_syn), "synthetic_rows disagrees with materialized table"
    print("ok  synthetic_rows matches materialized table")

    # 4. independent implementation: embedding_bag
    eb = F.embedding_bag(ids.reshape(-1, k), V.float(), per_sample_weights=w.reshape(-1, k),
                         mode="sum").reshape(T, h, WIDTH)
    assert torch.allclose(eb, y, atol=1e-5), f"embedding_bag max err {(eb - y).abs().max()}"
    print("ok  embedding_bag matches weighted_gather")

    # 5. chunking doesn't change the answer
    y_chunked = weighted_gather(ids, w, None, chunk_rows=k * 3)
    assert torch.allclose(y_chunked, y, atol=1e-6)
    print("ok  chunked weighted_gather matches unchunked")

    # 6. golden-file round trip
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        write_golden(tmp, ids, w, y, {"preset": "tiny", "seed": 1, "table": "synthetic"})
        gi, gw, gy, _ = load_golden(tmp)
        assert np.array_equal(gi, ids.reshape(-1, k).cpu().numpy())
        assert np.array_equal(gw, w.reshape(-1, k).cpu().numpy())
        assert np.array_equal(gy, y.reshape(-1, WIDTH).cpu().numpy())
    print("ok  golden files round-trip")
    print("all tests passed")


# --------------------------------------------------------------------------- CLI commands

def cmd_dump(args):
    T, h, d, n, k = PRESETS[args.preset]
    q, C1, C2 = make_inputs(T, h, d, n, seed=args.seed, device=args.device)
    y, ids, w = pkm_lookup(q, C1, C2, None, k)
    write_golden(args.out, ids, w, y, {"preset": args.preset, "seed": args.seed, "T": T, "h": h,
                                       "d": d, "n": n, "n_rows": n * n, "table": "synthetic"})


def cmd_replay(args):
    """Golden output for an id trace (e.g. from the Zipf workload generator)."""
    raw = np.fromfile(args.trace, dtype='<u4')
    if raw.size % args.k:
        raise SystemExit(f"trace length {raw.size} is not a multiple of k={args.k}")
    ids = torch.from_numpy(raw.astype(np.int64)).reshape(-1, args.k).to(args.device)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    w = torch.randn(ids.shape, generator=g).to(args.device).softmax(-1)
    y = weighted_gather(ids, w, None)
    write_golden(args.out, ids, w, y, {"trace": os.path.abspath(args.trace), "seed": args.seed,
                                       "table": "synthetic"})


def cmd_bench(args):
    """All-in-GPU PyTorch gather throughput (real fp16 table in device memory)."""
    dev = torch.device(args.device)
    if dev.type != "cuda":
        print("warning: not on CUDA; numbers are not meaningful for the GPU comparison")
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    V = torch.randn(args.rows, WIDTH, dtype=torch.float16, device=dev)
    N = args.lookups // args.k
    ids = torch.randint(0, args.rows, (N, args.k), device=dev)
    w = torch.rand(N, args.k, device=dev).softmax(-1)
    impls = {
        "weighted_gather": lambda: weighted_gather(ids, w, V),
        "embedding_bag": lambda: F.embedding_bag(ids, V, per_sample_weights=w.half(), mode="sum"),
    }
    for name, fn in impls.items():
        for _ in range(3):
            fn()
        sync()
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            fn()
            sync()
            times.append(time.perf_counter() - t0)
        t = float(np.median(times))
        rows = N * args.k
        print(f"{name:16s} {rows / t / 1e6:8.1f} M rows/s  {rows * WIDTH * 2 / t / 1e9:7.1f} GB/s  "
              f"({t * 1e3:.3f} ms median of {args.iters})")


def main():
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("test", help="run self-checks")
    t.add_argument("--device", default="cpu")

    d = sub.add_parser("dump", help="run pkm_lookup on random inputs and write golden files")
    d.add_argument("--preset", choices=PRESETS, default="small")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", required=True)
    d.add_argument("--device", default=default_dev)

    r = sub.add_parser("replay", help="golden files for an existing uint32 id trace")
    r.add_argument("--trace", required=True)
    r.add_argument("--k", type=int, default=32)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    r.add_argument("--device", default=default_dev)

    b = sub.add_parser("bench", help="PyTorch all-in-GPU gather throughput")
    b.add_argument("--rows", type=int, default=1 << 22)
    b.add_argument("--lookups", type=int, default=1 << 17)
    b.add_argument("--k", type=int, default=32)
    b.add_argument("--iters", type=int, default=20)
    b.add_argument("--device", default=default_dev)

    args = p.parse_args()
    if args.cmd == "test":
        run_tests(args.device)
    else:
        {"dump": cmd_dump, "replay": cmd_replay, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    main()
