# Project Plan — Rinha de Backend 2026

Fraud detection API in **Go** using a VP-Tree for vector similarity search.

---

## Challenge summary

Build a fraud detection HTTP API. For each incoming card transaction, vectorize it into 14 dimensions, find the 5 nearest neighbors in a 3M-record reference dataset, and return a fraud score and approval decision.

- Deadline: **2026-06-05T23:59:59-03:00**
- Test environment: Mac Mini Late 2014, 2.6 GHz, 8 GB RAM, Ubuntu 24.04
- Scoring: `final_score = score_p99 + score_det` (range: -6000 to +6000)

---

## Infrastructure constraints

| Constraint | Value |
|------------|-------|
| Total CPU across all services | ≤ 1 CPU |
| Total RAM across all services | ≤ 350 MB |
| Minimum topology | 1 load balancer + 2 API instances |
| Load balancer rule | Round-robin only, no business logic |
| Port | 9999 |
| Image platform | linux/amd64 |
| Network mode | bridge (no host, no privileged) |
| Branches | `main` (source), `submission` (docker-compose only) |
| License | MIT |

### Planned resource allocation

| Service | CPU | Memory |
|---------|-----|--------|
| nginx (LB) | 0.05 | 5 MB |
| api1 | 0.475 | 172 MB |
| api2 | 0.475 | 173 MB |
| **Total** | **1.0** | **350 MB** |

---

## API contract

### `GET /ready`
Returns `200 OK` only after the VP-Tree is fully built and the API is ready to serve requests. May take 20-60 seconds at startup — this is intentional and allowed.

### `POST /fraud-score`

**Request:**
```json
{
  "id": "tx-123",
  "transaction": { "amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z" },
  "customer": { "avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009"] },
  "merchant": { "id": "MERC-001", "mcc": "5912", "avg_amount": 298.95 },
  "terminal": { "is_online": false, "card_present": true, "km_from_home": 13.7 },
  "last_transaction": { "timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.86 }
}
```

**Response:**
```json
{ "approved": false, "fraud_score": 0.8 }
```

`approved = fraud_score < 0.6`
`fraud_score = count_of_fraud_among_5_nearest / 5`

---

## Vectorization — 14 dimensions

All values normalized to [0.0, 1.0] via `clamp(x) = min(max(x, 0.0), 1.0)`.
Exception: indices 5 and 6 use sentinel **-1** when `last_transaction` is null.

| idx | dimension | formula |
|-----|-----------|---------|
| 0 | amount | `clamp(transaction.amount / 10000)` |
| 1 | installments | `clamp(transaction.installments / 12)` |
| 2 | amount_vs_avg | `clamp((transaction.amount / customer.avg_amount) / 10)` |
| 3 | hour_of_day | `hour(requested_at UTC) / 23` |
| 4 | day_of_week | `weekday(requested_at UTC) / 6` (Mon=0, Sun=6) |
| 5 | minutes_since_last_tx | `clamp(minutes_diff / 1440)` or **-1** if null |
| 6 | km_from_last_tx | `clamp(last_transaction.km_from_current / 1000)` or **-1** if null |
| 7 | km_from_home | `clamp(terminal.km_from_home / 1000)` |
| 8 | tx_count_24h | `clamp(customer.tx_count_24h / 20)` |
| 9 | is_online | `1` if online else `0` |
| 10 | card_present | `1` if card present else `0` |
| 11 | unknown_merchant | `1` if merchant.id NOT in known_merchants else `0` |
| 12 | mcc_risk | lookup in mcc_risk table (default `0.5`) |
| 13 | merchant_avg_amount | `clamp(merchant.avg_amount / 10000)` |

### MCC risk table

| MCC | Risk |
|-----|------|
| 5411 | 0.15 |
| 5812 | 0.30 |
| 5912 | 0.20 |
| 5944 | 0.45 |
| 7801 | 0.80 |
| 7802 | 0.75 |
| 7995 | 0.85 |
| 4511 | 0.35 |
| 5311 | 0.25 |
| 5999 | 0.50 |

Unknown MCC → default `0.5`.

---

## Dataset

- `resources/references.json` — already decompressed, ~290 MB, 3,000,000 records
- Format: `{"vector": [14 floats], "label": "legit"|"fraud"}`
- Sentinel -1 is valid in positions 5 and 6 — must NOT be clamped or replaced
- File does not change during the test

---

## Chosen approach: VP-Tree (Vantage Point Tree)

### Why VP-Tree over alternatives

| Approach | Speed | Accuracy | Memory | Verdict |
|----------|-------|----------|--------|---------|
| Brute force | O(N×14) per query — too slow | Exact | Low | Rejected |
| VP-Tree | O(log N) per query | **Exact** | Low | **Chosen** |
| HNSW | O(log N) per query, faster constant | ~97% approximate | High (~200-500 MB) | Rejected — memory + approximation risk |
| External vector DB | Fast | Exact/Approx | Too high — overflows budget | Rejected |

HNSW was rejected because: (1) the graph structure for 3M vectors would consume ~200-500 MB, leaving no room for 2 API instances; (2) approximation errors translate directly into FP/FN detection penalties.

External databases (pgvector, Qdrant, SQLite-vss) were rejected because the raw 3M×14 float32 data alone is ~160 MB, and any DB engine adds significant overhead on top — overflowing the 350 MB total budget when combined with 2 API instances and a load balancer.

### Memory layout

Store everything as flat arrays (not pointer-based tree nodes) for cache efficiency:

```
vectors []float32   // 3M × 14 = 168 MB — arranged in BFS tree order
radii   []float32   // 3M × 4 bytes = 12 MB — split radius at each node
labels  []uint8     // 3M × 1 byte = 3 MB — 0=legit, 1=fraud
```

Tree structure is implicit: node at index `i` has children starting at `left(i)` and `right(i)`, computed from the build order. No pointers needed.

### Vantage point selection: sampled strategy

At each tree node, instead of picking a random vantage point:
1. Sample ~20 random candidates from the current partition
2. For each candidate, compute distances to another random sample of ~20 points
3. Pick the candidate with the highest variance of distances

This produces a more balanced tree that prunes more aggressively at query time. Construction is ~2-3× slower but query time drops by 30-50%. Worthwhile since construction is a one-time startup cost.

### Query: k=5 nearest neighbors, Euclidean distance

```
dist(a, b) = sqrt(sum((a[i] - b[i])^2 for i in 0..13))
```

The -1 sentinel values at indices 5 and 6 are handled naturally — a -1 in both query and reference means both had no prior transaction, and the distance between them will be ~0, which groups them correctly.

### Startup sequence (before `/ready` returns 200)

1. Parse `resources/references.json` — stream-parse to avoid peak memory spike
2. Build flat float32 arrays for vectors, labels
3. Build VP-Tree with sampled vantage point selection → compute radii
4. Re-arrange arrays in BFS order for cache-friendly access
5. Mark service as ready → `/ready` starts returning 200

Expected startup time: **20-60 seconds** (acceptable — test harness waits for `/ready`).

---

## Scoring strategy

### Latency (score_p99)

```
score_p99 = 1000 × log10(1000 / max(p99_ms, 1))
```

- p99 ≤ 1ms → +3000 (cap)
- p99 > 2000ms → -3000 (floor)
- Every 10× latency improvement = +1000 points

Targets: p99 < 10ms → +2000 pts, p99 < 1ms → +3000 pts.

### Detection (score_det)

```
E = 1×FP + 3×FN + 5×Err
failure_rate = (FP + FN + Err) / N
```

- failure_rate > 15% → -3000 (hard cutoff)
- Exact KNN (VP-Tree) means our detection matches the reference implementation exactly → minimal FP/FN

**Key rule:** Never return HTTP 500. On any internal error, return `{"approved": true, "fraud_score": 0.0}`. A false negative (weight 3) is always better than an HTTP error (weight 5) in the scoring formula.

---

## Project structure

```
rinha-de-backend-2026-NRLacerda/
├── cmd/
│   └── api/
│       └── main.go          # entry point
├── internal/
│   ├── vptree/
│   │   ├── tree.go          # VP-Tree build + query
│   │   └── tree_test.go
│   ├── vectorize/
│   │   ├── vectorize.go     # payload → [14]float32
│   │   └── vectorize_test.go
│   └── handler/
│       └── handler.go       # HTTP handlers
├── resources/
│   ├── references.json      # 3M reference vectors (already decompressed)
│   ├── mcc_risk.json
│   └── normalization.json
├── docs/
│   └── PROJECT-PLAN.md      # this file
├── Dockerfile
├── docker-compose.yml       # for local dev (not submission)
├── nginx.conf
├── go.mod
└── go.sum
```

### Submission branch structure

```
submission/
├── docker-compose.yml       # references pre-built public images
├── nginx.conf
└── info.json
```

---

## Implementation order

1. **`internal/vectorize`** — payload struct + vectorization function + tests
2. **`internal/vptree`** — dataset loading, tree build, k=5 query + tests
3. **`internal/handler`** — `GET /ready`, `POST /fraud-score`
4. **`cmd/api/main.go`** — wire everything, startup sequence
5. **`Dockerfile`** — multi-stage build, embed resources in image or mount
6. **`nginx.conf`** — round-robin across api1:8080 and api2:8080
7. **`docker-compose.yml`** — resource limits, port 9999, healthcheck on `/ready`
8. **Tune** — measure p99, adjust vantage point sampling, run k6 locally

---

## Rules summary

- Load balancer must not apply any detection logic
- Do not use test payloads as reference data for lookup
- All images must be public and linux/amd64 compatible
- Repository must be MIT licensed and public
- `submission` branch contains only docker-compose.yml and config files (no source)
- Register by opening a PR to the main rinha repo adding `participants/NRLacerda.json`
- Trigger a test by opening an issue with `rinha/test` in the description
