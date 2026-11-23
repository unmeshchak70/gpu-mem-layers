# Microbenchmark summary

Runs combined (median): results/ghc33.csv

## Peak bandwidth (largest size measured)

| Path | Bench | GB/s |
|---|---|---|
| Device sequential read | `dev_seq` | 425.90 |
| Device gather, read only | `dev_gather_ro` | 416.98 |
| Device gather + write | `dev_gather` | 195.19 |
| Zero-copy sequential read | `zc_seq` | 12.86 |
| Zero-copy gather, read only | `zc_gather_ro` | 12.85 |
| cudaMemcpy H2D, pinned | `h2d_pinned` | 12.94 |
| cudaMemcpy H2D, pageable | `h2d_pageable` | 10.46 |
| Bulk fetch total | `bulk_total` | 5.62 |
| 1 KB cudaMemcpy per row | `h2d_per_row` | 0.41 |

Zero-copy gather reaches 90% of its peak at m = 1K rows.
Per-row cudaMemcpy costs 2.51 µs per row.
Bulk fetch peaks at 7.08 GB/s at m = 2K rows.

## Zero-copy vs. bulk fetch (median ms)

| m rows | zero-copy | bulk total | bulk CPU pack | bulk copy + scatter | bulk ÷ zc |
|---|---|---|---|---|---|
| 1 | 0.0074 | 0.0262 | 0.0008 | 0.0188 | 3.55 |
| 4 | 0.0066 | 0.0265 | 0.0011 | 0.0200 | 4.03 |
| 16 | 0.0075 | 0.0291 | 0.0031 | 0.0201 | 3.90 |
| 64 | 0.0111 | 0.0373 | 0.0112 | 0.0201 | 3.36 |
| 256 | 0.0267 | 0.0566 | 0.0137 | 0.0368 | 2.12 |
| 1K | 0.0876 | 0.1628 | 0.0534 | 0.1035 | 1.86 |
| 4K | 0.3334 | 0.6109 | 0.2465 | 0.3553 | 1.83 |
| 16K | 1.3124 | 2.7399 | 1.3286 | 1.3988 | 2.09 |
| 64K | 5.2306 | 11.7162 | 6.1464 | 5.5598 | 2.24 |
| 256K | 20.8957 | 47.6224 | 25.3979 | 22.2098 | 2.28 |
| 1M | 83.5405 | 191.0729 | 102.3447 | 88.7201 | 2.29 |

## Performance model

B_device = 417.0 GB/s (`dev_gather_ro`), B_fetch = 12.85 GB/s (`zc_gather_ro`), r = 32.4.
Slowdown S(h) = h + r(1−h); staying within 5× needs read hit rate h ≥ 0.873.

## Predicted slowdown on synthetic traces (LRU)

Proposal model uses the read hit rate. Dedup-aware: S = 1 + r·u·(1 − h_unique).

| trace | mode | α | drift | cache | read hit | unique hit | unique/ids | S (model) | S (dedup-aware) |
|---|---|---|---|---|---|---|---|---|---|
| long_a0.6 | iid | 0.6 | 0.01 | 1% | 0.040 | 0.028 | 0.928 | 31.2× | 30.2× |
| long_a0.6 | iid | 0.6 | 0.01 | 5% | 0.164 | 0.127 | 0.928 | 27.3× | 27.3× |
| long_a0.6 | iid | 0.6 | 0.01 | 10% | 0.255 | 0.216 | 0.928 | 24.4× | 24.6× |
| long_a0.6 | iid | 0.6 | 0.01 | 25% | 0.447 | 0.414 | 0.928 | 18.4× | 18.6× |
| long_a0.6 | iid | 0.6 | 0.01 | 50% | 0.674 | 0.653 | 0.928 | 11.2× | 11.4× |
| long_a0.8 | iid | 0.8 | 0.01 | 1% | 0.127 | 0.060 | 0.754 | 28.5× | 24.0× |
| long_a0.8 | iid | 0.8 | 0.01 | 5% | 0.395 | 0.229 | 0.754 | 20.0× | 19.9× |
| long_a0.8 | iid | 0.8 | 0.01 | 10% | 0.490 | 0.342 | 0.754 | 17.1× | 17.1× |
| long_a0.8 | iid | 0.8 | 0.01 | 25% | 0.651 | 0.546 | 0.754 | 12.0× | 12.1× |
| long_a0.8 | iid | 0.8 | 0.01 | 50% | 0.800 | 0.740 | 0.754 | 7.3× | 7.4× |
| long_a1.0 | iid | 1 | 0.01 | 1% | 0.430 | 0.157 | 0.420 | 18.9× | 12.5× |
| long_a1.0 | iid | 1 | 0.01 | 5% | 0.735 | 0.401 | 0.420 | 9.3× | 9.2× |
| long_a1.0 | iid | 1 | 0.01 | 10% | 0.789 | 0.521 | 0.420 | 7.6× | 7.5× |
| long_a1.0 | iid | 1 | 0.01 | 25% | 0.865 | 0.693 | 0.420 | 5.3× | 5.2× |
| long_a1.0 | iid | 1 | 0.01 | 50% | 0.908 | 0.793 | 0.420 | 3.9× | 3.8× |
| long_a1.2 | iid | 1.2 | 0.01 | 1% | 0.894 | 0.398 | 0.149 | 4.3× | 3.9× |
| long_a1.2 | iid | 1.2 | 0.01 | 5% | 0.935 | 0.638 | 0.149 | 3.1× | 2.7× |
| long_a1.2 | iid | 1.2 | 0.01 | 10% | 0.947 | 0.720 | 0.149 | 2.7× | 2.3× |
| long_a1.2 | iid | 1.2 | 0.01 | 25% | 0.955 | 0.772 | 0.149 | 2.4× | 2.1× |
| long_a1.2 | iid | 1.2 | 0.01 | 50% | 0.955 | 0.772 | 0.149 | 2.4× | 2.1× |
