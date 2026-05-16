# Tech Stack

Every choice here optimizes for raw request latency (p99) over ergonomics.

---

## HTTP server — `valyala/fasthttp`

No framework (no Fiber, no Echo). Raw `fasthttp` directly.

- Worker pool: reuses goroutines instead of spawning one per request
- `RequestCtx` pooled via `sync.Pool` — zero allocations per request on the HTTP layer
- No `net/http` compatibility layer overhead
- ~5-10× higher throughput than `net/http` in benchmarks

```go
fasthttp.ListenAndServe(":8080", handler)
```

---

## JSON parsing (hot path) — `bytedance/sonic`

Used only for decoding the `POST /fraud-score` request body.

- SIMD-accelerated (AVX2) on linux/amd64 — the test environment (Mac Mini 2014, Intel Haswell) supports AVX2
- ~3-5× faster than `encoding/json`
- CGo-based; the Dockerfile build stage includes `gcc`
- Drop-in `sonic.Unmarshal` / `sonic.Marshal`

The dataset loader (`cmd/build-index`) uses stdlib `encoding/json` streaming — it runs once at Docker build time, speed there is irrelevant.

---

## VP-Tree binary index — `encoding/binary` (stdlib)

The VP-Tree is built once at Docker image build time by `cmd/build-index`, serialized as a raw binary file (`resources/vptree.bin`), and embedded into the image.

At runtime, the API reads the binary file directly into flat `[]float32` / `[]uint8` arrays. No JSON parsing, no tree construction at runtime.

Binary layout (little-endian):

```
[4 bytes] N          — number of nodes (int32)
[N × 14 × 4 bytes]   — vectors (float32, BFS order)
[N × 4 bytes]        — radii (float32)
[N × 1 byte]         — labels (0=legit, 1=fraud)
```

Reading ~183 MB of flat binary: ~1-2 seconds. Vs building from JSON: ~30-60 seconds.

---

## Distance computation — stdlib `math` (mostly avoided)

Euclidean distance is computed as **squared distance** throughout the entire VP-Tree search. `math.Sqrt` is never called during KNN queries — we only compare distances, never return them. Sqrt is skipped entirely.

The 14-dimensional dot product is **manually unrolled** — no loop, 14 explicit multiply-add pairs. At 14 dims the compiler can emit SIMD with unrolled code; a loop may not trigger auto-vectorization.

---

## KNN heap — inline max-heap (no library)

`k=5` is fixed. The heap is a 5-element `[5]float32` array of squared distances + a parallel `[5]uint8` label array, maintained as a max-heap inline in the query function. No `container/heap` interface overhead (avoids `interface{}` boxing on every comparison).

When the heap is full, `heap[0]` (the max) is the pruning threshold for VP-Tree branch elimination.

---

## GOMAXPROCS — set to 1 at startup

Each API instance is limited to `0.475` CPU. Running with `GOMAXPROCS > 1` creates scheduler contention for no throughput gain. Set explicitly:

```go
runtime.GOMAXPROCS(1)
```

---

## `sync.Pool` — request struct reuse

The request payload struct (`FraudRequest`) and the intermediate `[14]float32` vector are pooled to avoid per-request heap allocation and GC pressure under sustained load.

---

## Summary

| Concern | Choice | Why |
|---------|--------|-----|
| HTTP | `valyala/fasthttp` | Zero-alloc worker pool |
| JSON decode | `bytedance/sonic` | SIMD, 3-5× stdlib |
| JSON encode | `bytedance/sonic` | Consistent, fast |
| VP-Tree storage | flat `[]float32` BFS binary | Cache-friendly, zero parsing overhead |
| Distance | squared Euclidean, unrolled | No sqrt, no loop overhead |
| KNN heap | inline `[5]` arrays | No interface boxing |
| Concurrency | `GOMAXPROCS=1` | Matches CPU quota |
| Alloc reduction | `sync.Pool` | Reduces GC pauses |
| Dataset build | `encoding/json` + `encoding/binary` | Build-time only, irrelevant speed |

---

## go.mod dependencies (external)

```
github.com/valyala/fasthttp
github.com/bytedance/sonic
```

Everything else is Go stdlib.
