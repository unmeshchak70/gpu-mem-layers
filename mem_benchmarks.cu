// Memory microbenchmarks for the product-key memory buffer pool.
//
// Measures the bandwidths in the performance model B_eff = (h/B_device + (1-h)/B_fetch)^-1
// and the miss count m* where a bulk fetch starts to beat zero-copy reads.
//
// Build (RTX 2080 = Turing, sm_75):
//   nvcc -O3 -std=c++17 -arch=sm_75 -Xcompiler -fopenmp -o mem_benchmarks mem_benchmarks.cu
// Run:
//   ./mem_benchmarks > results.csv                 # everything (~3 GB pinned host, ~4.2 GB device)
//   ./mem_benchmarks --only zc_gather,bulk_gather  # subset
//   ./mem_benchmarks --help
//
// Output: CSV on stdout, progress and correctness checks on stderr.
//   bench,bytes,rows,time_ms,gbps,rows_per_s
//   bytes = useful bytes moved (rows * 1 KB for gathers); gbps = bytes / time.
//   Every time is the median of --iters runs after 3 warmup runs.
//
// Benchmarks (1 KB rows, ids uniform over the table, a fresh id set every run so
// results are not L2 hits):
//   dev_seq        sequential read of device memory                       4 KB .. max-bytes
//   dev_gather     warp-per-row gather, device table -> device buffer     m = 1 .. max-rows
//   dev_gather_ro  same, read only (what the weighted sum does on a hit)
//   zc_seq         sequential read of mapped pinned host memory (zero-copy)
//   zc_gather      warp-per-row gather, mapped host table -> device buffer
//   zc_gather_ro   same, read only (zero-copy miss path feeding the weighted sum)
//   h2d_pinned     one cudaMemcpyAsync from pinned host memory            4 KB .. max-bytes
//   h2d_pageable   one cudaMemcpyAsync from malloc'd host memory
//   h2d_per_row    one 1 KB cudaMemcpyAsync per row (naive on-demand fetch) m = 1 .. max-per-row
//   bulk_gather    CPU copies m rows into pinned staging, one H2D copy, scatter kernel
//                  into random cache slots. Emits bulk_cpu, bulk_xfer (copy + scatter)
//                  and bulk_total rows. Reads the same pinned table as zc_gather.
//
// Table contents: 32-bit word i of a table holds the value i, so word j of row r is
// r*256 + j. Gathers that write output are checked against this after each sweep.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

constexpr size_t ROW_BYTES = 1024;                       // 512 fp16 values
constexpr int VEC_PER_ROW = ROW_BYTES / sizeof(uint4);   // 64 x 16 B
constexpr int WORDS_PER_ROW = ROW_BYTES / sizeof(uint32_t);
constexpr int THREADS = 256;                             // 8 warps per block
constexpr unsigned MAGIC = 0x9E3779B9u;

// --------------------------------------------------------------------------- kernels

__global__ void fill_index(uint32_t* p, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = (uint32_t)i;
}

// Sequential read. The data-dependent store keeps the compiler from dropping the loads.
__global__ void read_seq(const uint4* __restrict__ p, size_t n, unsigned* sink) {
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == MAGIC) *sink = acc.x;
}

// One warp per 1 KB row. Lane l loads 16 B at l and l+32, so each load instruction of
// the warp covers 512 contiguous bytes. `table` may be device memory or a mapped host
// pointer (zero-copy); the kernel is identical.
template <bool WRITE>
__global__ void gather_rows(const uint4* __restrict__ table, const uint32_t* __restrict__ ids,
                            int m, uint4* __restrict__ out, unsigned* sink) {
  int w = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (w >= m) return;
  const uint4* row = table + (size_t)ids[w] * VEC_PER_ROW;
  uint4 a = row[lane];
  uint4 b = row[lane + 32];
  if (WRITE) {
    uint4* dst = out + (size_t)w * VEC_PER_ROW;
    dst[lane] = a;
    dst[lane + 32] = b;
  } else if ((a.x ^ a.y ^ b.z ^ b.w) == MAGIC) {
    *sink = a.x;
  }
}

// Last step of the bulk path: staged row w goes to cache slot slots[w].
__global__ void scatter_rows(const uint4* __restrict__ staging, const uint32_t* __restrict__ slots,
                             int m, uint4* __restrict__ cache) {
  int w = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (w >= m) return;
  const uint4* src = staging + (size_t)w * VEC_PER_ROW;
  uint4* dst = cache + (size_t)slots[w] * VEC_PER_ROW;
  dst[lane] = src[lane];
  dst[lane + 32] = src[lane + 32];
}

static inline int warp_blocks(int m) { return (m + THREADS / 32 - 1) / (THREADS / 32); }

// --------------------------------------------------------------------------- harness

struct Args {
  size_t dev_rows = size_t(1) << 21;   // device table: 2 GB
  size_t host_rows = size_t(1) << 21;  // pinned host table: 2 GB
  size_t max_bytes = size_t(1) << 30;  // largest sequential / memcpy size
  int max_rows = 1 << 20;              // largest gather (rows)
  int max_per_row = 1 << 14;           // largest h2d_per_row sweep (it is slow)
  int iters = 20;
  std::string only;                    // comma-separated bench names; empty = all
};

static void usage(int code) {
  fprintf(code ? stderr : stdout,
          "usage: mem_benchmarks [--dev-rows N] [--host-rows N] [--max-rows N] [--max-bytes N]\n"
          "                      [--max-per-row N] [--iters N] [--only a,b,...]\n"
          "benches: dev_seq dev_gather dev_gather_ro zc_seq zc_gather zc_gather_ro\n"
          "         h2d_pinned h2d_pageable h2d_per_row bulk_gather\n");
  exit(code);
}

static Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) usage(1);
      return argv[++i];
    };
    if (k == "--dev-rows") a.dev_rows = strtoull(next(), nullptr, 0);
    else if (k == "--host-rows") a.host_rows = strtoull(next(), nullptr, 0);
    else if (k == "--max-bytes") a.max_bytes = strtoull(next(), nullptr, 0);
    else if (k == "--max-rows") a.max_rows = (int)strtol(next(), nullptr, 0);
    else if (k == "--max-per-row") a.max_per_row = (int)strtol(next(), nullptr, 0);
    else if (k == "--iters") a.iters = (int)strtol(next(), nullptr, 0);
    else if (k == "--only") a.only = next();
    else if (k == "-h" || k == "--help") usage(0);
    else usage(1);
  }
  if (a.iters < 1 || a.max_rows < 1 || a.dev_rows < 1 || a.host_rows < 1) usage(1);
  if (a.dev_rows > (size_t(1) << 32) || a.host_rows > (size_t(1) << 32)) {
    fprintf(stderr, "row ids are uint32: tables must have <= 2^32 rows\n");
    exit(1);
  }
  a.max_per_row = std::min(a.max_per_row, a.max_rows);
  return a;
}

static bool want(const Args& a, const char* name) {
  return a.only.empty() || ("," + a.only + ",").find(std::string(",") + name + ",") != std::string::npos;
}

static float median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Median GPU time (cudaEvents on the default stream) of fn(i) over iters runs.
// Warmup runs call fn(iters), fn(iters+1), fn(iters+2), so every call sees its own id set.
template <class F>
static float median_ms(int iters, F fn) {
  static cudaEvent_t a = nullptr, b = nullptr;
  if (!a) {
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
  }
  for (int w = 0; w < 3; w++) fn(iters + w);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<float> t(iters);
  for (int i = 0; i < iters; i++) {
    CK(cudaEventRecord(a));
    fn(i);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&t[i], a, b));
  }
  CK(cudaGetLastError());
  return median(t);
}

static void csv(const char* bench, size_t bytes, size_t rows, double ms) {
  double s = ms / 1e3;
  printf("%s,%zu,%zu,%.6f,%.3f,%.1f\n", bench, bytes, rows, ms, bytes / s / 1e9, rows / s);
  fflush(stdout);
}

struct Bufs {
  int num_sms = 0;
  int nsets = 0;                       // iters + 3 id sets of max_rows ids each

  uint4* d_table = nullptr;            // device table, dev_rows rows
  size_t d_table_bytes = 0;
  uint4* h_table = nullptr;            // pinned + mapped host table, host_rows rows
  const uint4* h_table_dev = nullptr;  // device-side alias of h_table (zero-copy)
  size_t h_table_bytes = 0;
  void* h_pageable = nullptr;          // malloc'd source for h2d_pageable
  size_t h_pageable_bytes = 0;

  uint4* d_out = nullptr;              // gather / memcpy destination, also the "cache"
  size_t d_out_bytes = 0;
  uint4* h_staging = nullptr;          // pinned staging for bulk_gather, max_rows rows
  uint4* d_staging = nullptr;

  std::vector<uint32_t> h_ids_dev, h_ids_host;  // [nsets][max_rows]
  uint32_t* d_ids_dev = nullptr;                // ids over dev_rows
  uint32_t* d_ids_host = nullptr;               // ids over host_rows
  std::vector<uint32_t> h_slots;                // random permutation of [0, max_rows)
  uint32_t* d_slots = nullptr;
  unsigned* d_sink = nullptr;
};

// Compare a sample of output rows against the table pattern (word j of row r = r*256 + j).
static void check_rows(const char* name, const uint4* d_out, const uint32_t* ids,
                       const uint32_t* slots, int m) {
  std::vector<uint32_t> row(WORDS_PER_ROW);
  int step = std::max(1, m / 64);
  for (int i = 0; i < m; i += step) {
    size_t dst = slots ? slots[i] : (size_t)i;
    CK(cudaMemcpy(row.data(), d_out + dst * VEC_PER_ROW, ROW_BYTES, cudaMemcpyDeviceToHost));
    for (int j = 0; j < WORDS_PER_ROW; j++) {
      uint32_t want_v = (uint32_t)((size_t)ids[i] * WORDS_PER_ROW + j);
      if (row[j] != want_v) {
        fprintf(stderr, "FAIL %s: output row %zu (table row %u) word %d = %u, want %u\n", name,
                dst, ids[i], j, row[j], want_v);
        exit(1);
      }
    }
  }
  fprintf(stderr, "check %-14s ok\n", name);
}

// --------------------------------------------------------------------------- benchmarks

static void bench_seq(const char* name, const uint4* base, size_t avail, const Args& a,
                      const Bufs& B) {
  size_t top = std::min(a.max_bytes, avail);
  for (size_t bytes = 4096; bytes <= top; bytes *= 2) {
    size_t n = bytes / sizeof(uint4);
    size_t windows = avail / bytes;  // non-overlapping windows; rotate through them
    int blocks = (int)std::min<size_t>((size_t)B.num_sms * 8, (n + THREADS - 1) / THREADS);
    float ms = median_ms(a.iters, [&](int i) {
      read_seq<<<blocks, THREADS>>>(base + (size_t)(i % windows) * n, n, B.d_sink);
    });
    csv(name, bytes, bytes / ROW_BYTES, ms);
  }
}

template <bool WRITE>
static void bench_gather(const char* name, const uint4* table, const uint32_t* d_ids,
                         const std::vector<uint32_t>& h_ids, const Args& a, const Bufs& B) {
  for (int m = 1; m <= a.max_rows; m *= 2) {
    float ms = median_ms(a.iters, [&](int i) {
      gather_rows<WRITE><<<warp_blocks(m), THREADS>>>(table, d_ids + (size_t)i * a.max_rows, m,
                                                       B.d_out, B.d_sink);
    });
    csv(name, (size_t)m * ROW_BYTES, m, ms);
  }
  if (WRITE) {
    int m = std::min(a.max_rows, 4096);
    gather_rows<true><<<warp_blocks(m), THREADS>>>(table, d_ids, m, B.d_out, B.d_sink);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    check_rows(name, B.d_out, h_ids.data(), nullptr, m);
  }
}

static void bench_h2d(const char* name, const void* src, size_t avail, const Args& a,
                      const Bufs& B) {
  size_t top = std::min({a.max_bytes, avail, B.d_out_bytes});
  for (size_t bytes = 4096; bytes <= top; bytes *= 2) {
    size_t windows = avail / bytes;
    float ms = median_ms(a.iters, [&](int i) {
      CK(cudaMemcpyAsync(B.d_out, (const char*)src + (size_t)(i % windows) * bytes, bytes,
                         cudaMemcpyHostToDevice));
    });
    csv(name, bytes, bytes / ROW_BYTES, ms);
  }
}

static void per_row_copy(const Bufs& B, const uint32_t* ids, int m) {
  for (int r = 0; r < m; r++)
    CK(cudaMemcpyAsync(B.d_out + (size_t)r * VEC_PER_ROW, B.h_table + (size_t)ids[r] * VEC_PER_ROW,
                       ROW_BYTES, cudaMemcpyHostToDevice));
}

static void bench_per_row(const Args& a, const Bufs& B) {
  for (int m = 1; m <= a.max_per_row; m *= 2) {
    float ms = median_ms(a.iters, [&](int i) {
      per_row_copy(B, B.h_ids_host.data() + (size_t)i * a.max_rows, m);
    });
    csv("h2d_per_row", (size_t)m * ROW_BYTES, m, ms);
  }
  int m = std::min(a.max_per_row, 4096);
  per_row_copy(B, B.h_ids_host.data(), m);
  CK(cudaDeviceSynchronize());
  check_rows("h2d_per_row", B.d_out, B.h_ids_host.data(), nullptr, m);
}

static void cpu_gather(uint4* dst, const uint4* table, const uint32_t* ids, int m) {
#pragma omp parallel for schedule(static) if (m >= 256)
  for (int i = 0; i < m; i++)
    memcpy(dst + (size_t)i * VEC_PER_ROW, table + (size_t)ids[i] * VEC_PER_ROW, ROW_BYTES);
}

// CPU gather -> one H2D copy -> scatter into cache slots. Returns {cpu, xfer, total} ms.
static void bulk_once(const Bufs& B, const uint32_t* ids, int m, cudaEvent_t e0, cudaEvent_t e1,
                      float* cpu_ms, float* xfer_ms, float* total_ms) {
  using clk = std::chrono::steady_clock;
  auto t0 = clk::now();
  cpu_gather(B.h_staging, B.h_table, ids, m);
  auto t1 = clk::now();
  CK(cudaEventRecord(e0));
  CK(cudaMemcpyAsync(B.d_staging, B.h_staging, (size_t)m * ROW_BYTES, cudaMemcpyHostToDevice));
  scatter_rows<<<warp_blocks(m), THREADS>>>(B.d_staging, B.d_slots, m, B.d_out);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  auto t2 = clk::now();
  CK(cudaGetLastError());
  CK(cudaEventElapsedTime(xfer_ms, e0, e1));
  *cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
  *total_ms = std::chrono::duration<float, std::milli>(t2 - t0).count();
}

static void bench_bulk(const Args& a, const Bufs& B) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int m = 1; m <= a.max_rows; m *= 2) {
    std::vector<float> tc, tx, tt;
    for (int it = -3; it < a.iters; it++) {
      int set = it < 0 ? a.iters + 3 + it : it;  // warmups use sets iters..iters+2
      float c, x, t;
      bulk_once(B, B.h_ids_host.data() + (size_t)set * a.max_rows, m, e0, e1, &c, &x, &t);
      if (it < 0) continue;
      tc.push_back(c);
      tx.push_back(x);
      tt.push_back(t);
    }
    size_t bytes = (size_t)m * ROW_BYTES;
    csv("bulk_cpu", bytes, m, median(tc));
    csv("bulk_xfer", bytes, m, median(tx));
    csv("bulk_total", bytes, m, median(tt));
  }
  int m = std::min(a.max_rows, 4096);
  float c, x, t;
  bulk_once(B, B.h_ids_host.data(), m, e0, e1, &c, &x, &t);
  check_rows("bulk_gather", B.d_out, B.h_ids_host.data(), B.h_slots.data(), m);
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
}

// --------------------------------------------------------------------------- setup

static void make_ids(std::vector<uint32_t>& h, uint32_t** d, size_t n, size_t rows, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint64_t> dist(0, rows - 1);
  h.resize(n);
  for (auto& x : h) x = (uint32_t)dist(rng);
  CK(cudaMalloc(d, n * sizeof(uint32_t)));
  CK(cudaMemcpy(*d, h.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  CK(cudaSetDeviceFlags(cudaDeviceMapHost));  // before the context is created

  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  fprintf(stderr, "device: %s (sm_%d%d), %d SMs, %.1f GB, L2 %d KB, canMapHostMemory=%d\n",
          p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem / 1e9,
          p.l2CacheSize / 1024, p.canMapHostMemory);
  if (!p.canMapHostMemory) {
    fprintf(stderr, "device cannot map host memory; zero-copy benchmarks impossible\n");
    return 1;
  }

  Bufs B;
  B.num_sms = p.multiProcessorCount;
  B.nsets = a.iters + 3;
  bool need_dev = want(a, "dev_seq") || want(a, "dev_gather") || want(a, "dev_gather_ro");
  bool need_host = want(a, "zc_seq") || want(a, "zc_gather") || want(a, "zc_gather_ro") ||
                   want(a, "h2d_pinned") || want(a, "h2d_per_row") || want(a, "bulk_gather");
  auto t_setup = std::chrono::steady_clock::now();

  B.d_out_bytes = std::max((size_t)a.max_rows * ROW_BYTES, a.max_bytes);
  CK(cudaMalloc(&B.d_out, B.d_out_bytes));
  CK(cudaMalloc(&B.d_sink, sizeof(unsigned)));

  if (need_dev) {
    B.d_table_bytes = a.dev_rows * ROW_BYTES;
    fprintf(stderr, "allocating device table: %.2f GB\n", B.d_table_bytes / 1e9);
    CK(cudaMalloc(&B.d_table, B.d_table_bytes));
    fill_index<<<B.num_sms * 8, THREADS>>>((uint32_t*)B.d_table, B.d_table_bytes / 4);
    CK(cudaGetLastError());
    make_ids(B.h_ids_dev, &B.d_ids_dev, (size_t)B.nsets * a.max_rows, a.dev_rows, 1);
  }

  if (need_host) {
    B.h_table_bytes = a.host_rows * ROW_BYTES;
    fprintf(stderr, "allocating pinned host table: %.2f GB\n", B.h_table_bytes / 1e9);
    CK(cudaHostAlloc(&B.h_table, B.h_table_bytes, cudaHostAllocMapped));
    void* dp = nullptr;
    CK(cudaHostGetDevicePointer(&dp, B.h_table, 0));
    B.h_table_dev = (const uint4*)dp;
    uint32_t* w = (uint32_t*)B.h_table;
    long long nw = (long long)(B.h_table_bytes / 4);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < nw; i++) w[i] = (uint32_t)i;
    make_ids(B.h_ids_host, &B.d_ids_host, (size_t)B.nsets * a.max_rows, a.host_rows, 2);
  }

  if (want(a, "bulk_gather")) {
    CK(cudaHostAlloc(&B.h_staging, (size_t)a.max_rows * ROW_BYTES, cudaHostAllocDefault));
    CK(cudaMalloc(&B.d_staging, (size_t)a.max_rows * ROW_BYTES));
    B.h_slots.resize(a.max_rows);
    for (int i = 0; i < a.max_rows; i++) B.h_slots[i] = (uint32_t)i;
    std::mt19937_64 rng(3);
    std::shuffle(B.h_slots.begin(), B.h_slots.end(), rng);
    CK(cudaMalloc(&B.d_slots, (size_t)a.max_rows * sizeof(uint32_t)));
    CK(cudaMemcpy(B.d_slots, B.h_slots.data(), (size_t)a.max_rows * sizeof(uint32_t),
                  cudaMemcpyHostToDevice));
  }

  if (want(a, "h2d_pageable")) {
    B.h_pageable_bytes = a.max_bytes;
    B.h_pageable = malloc(B.h_pageable_bytes);
    if (!B.h_pageable) {
      fprintf(stderr, "malloc of %zu bytes failed\n", B.h_pageable_bytes);
      return 1;
    }
    memset(B.h_pageable, 1, B.h_pageable_bytes);  // fault the pages in
  }

  CK(cudaDeviceSynchronize());
  fprintf(stderr, "setup: %.1f s\n",
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t_setup).count());

  printf("bench,bytes,rows,time_ms,gbps,rows_per_s\n");
  auto run = [&](const char* name, auto fn) {
    if (!want(a, name)) return;
    fprintf(stderr, "running %s\n", name);
    fn();
  };
  run("dev_seq", [&] { bench_seq("dev_seq", B.d_table, B.d_table_bytes, a, B); });
  run("dev_gather", [&] { bench_gather<true>("dev_gather", B.d_table, B.d_ids_dev, B.h_ids_dev, a, B); });
  run("dev_gather_ro", [&] { bench_gather<false>("dev_gather_ro", B.d_table, B.d_ids_dev, B.h_ids_dev, a, B); });
  run("zc_seq", [&] { bench_seq("zc_seq", B.h_table_dev, B.h_table_bytes, a, B); });
  run("zc_gather", [&] { bench_gather<true>("zc_gather", B.h_table_dev, B.d_ids_host, B.h_ids_host, a, B); });
  run("zc_gather_ro", [&] { bench_gather<false>("zc_gather_ro", B.h_table_dev, B.d_ids_host, B.h_ids_host, a, B); });
  run("h2d_pinned", [&] { bench_h2d("h2d_pinned", B.h_table, B.h_table_bytes, a, B); });
  run("h2d_pageable", [&] { bench_h2d("h2d_pageable", B.h_pageable, B.h_pageable_bytes, a, B); });
  run("h2d_per_row", [&] { bench_per_row(a, B); });
  run("bulk_gather", [&] { bench_bulk(a, B); });

  if (B.d_table) CK(cudaFree(B.d_table));
  if (B.h_table) CK(cudaFreeHost(B.h_table));
  if (B.h_staging) CK(cudaFreeHost(B.h_staging));
  if (B.d_staging) CK(cudaFree(B.d_staging));
  if (B.d_ids_dev) CK(cudaFree(B.d_ids_dev));
  if (B.d_ids_host) CK(cudaFree(B.d_ids_host));
  if (B.d_slots) CK(cudaFree(B.d_slots));
  CK(cudaFree(B.d_out));
  CK(cudaFree(B.d_sink));
  free(B.h_pageable);
  return 0;
}
