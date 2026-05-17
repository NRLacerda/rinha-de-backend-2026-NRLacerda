# Rinha de Backend 2026 — NRLacerda

Fraud detection API built for the [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) challenge.

## What it does

Receives a POST `/fraud-score` with transaction data, runs a 5-nearest-neighbor search against 3 million labeled reference vectors, and returns a fraud score + approval decision in under 2 seconds.

```
POST /fraud-score
{"approved": true, "fraud_score": 0.2}
```

## Architecture

```
nginx (LB, port 9999)
  ├── api1 (port 8080)  ─┐
  └── api2 (port 8080)  ─┴─► index (port 9000)
```

The challenge mandates exactly this topology: 1 nginx, 2 API replicas, 1 index service.

- **nginx** — round-robin load balancer, minimal config, 0.05 CPU / 6 MB
- **api1/api2** — HTTP handlers that vectorize the request and proxy to index; 0.2 CPU / 12 MB each
- **index** — holds the HNSW graph in memory, answers KNN queries; 0.55 CPU / 316 MB

Total: 1.0 CPU / 350 MB.

## How the index works

### Dataset

3 million reference transactions, each labeled 0 (legit) or 1 (fraud). Stored as uint8-quantized 14-dimensional vectors. The index is pre-built at Docker image build time and baked into the image as `resources/hnsw.bin`.

### Feature vector (14 dims)

| # | Feature | Normalization |
|---|---------|---------------|
| 0 | amount | / 10 000, clamped [0,1] |
| 1 | installments | / 12 |
| 2 | amount vs customer avg | ratio / 10 |
| 3 | hour of day | / 23 |
| 4 | day of week | Mon=0, Sun=6, / 6 |
| 5 | minutes since last tx | / 1440 (−1 if no last tx) |
| 6 | km from last tx | / 1000 (−1 if no last tx) |
| 7 | km from home | / 1000 |
| 8 | tx count 24h | / 20 |
| 9 | is online | 0/1 |
| 10 | card present | 0/1 |
| 11 | unknown merchant | 1 if not in known list |
| 12 | MCC risk score | lookup table, default 0.5 |
| 13 | merchant avg amount | / 10 000 |

### HNSW graph

Custom HNSW implementation (`internal/hnsw`) with uint8-quantized vectors and squared integer Euclidean distance (no sqrt needed for ranking).

Key parameters:
- **M=8, M0=16** — graph connectivity (upper layers / level-0)
- **efConstruction=200** — build quality
- **efSearch=6** — query candidate pool; recall ≈ 98.5% on this 14-dim dataset, same as ef=200

The fraud score for a query is `FraudCount(5-NN) / 5.0`.

### Memory layout (3M vectors, M=8)

| Structure | Size |
|-----------|------|
| vectors `[N×14]uint8` | 42 MB |
| conn0 `[N×M0]int32` | 192 MB |
| conn0cnt `[N]uint8` | 3 MB |
| upperConns (sparse map) | ~30 MB |
| visitSlot pool (×2) | 24 MB |
| runtime / misc | ~15 MB |
| **Total** | **~306 MB** |

### Concurrent queries

HNSW graph traversal requires per-query visited markers to avoid revisiting nodes. Instead of a shared array (which would serialize all queries), each query borrows a `visitSlot` from a bounded channel pool:

```go
vs := <-h.visitPool        // borrow
defer func() { h.visitPool <- vs }()  // return
```

Each slot uses a generation counter trick: incrementing `gen` marks a new query epoch without clearing the 12 MB array on every call. Two slots = 2 concurrent HNSW queries, which saturates the 0.55 CPU budget at efSearch=6.

### Startup

The index loads `hnsw.bin` on start (~7 minutes at build time, ~2–3 seconds at runtime). The API containers poll `GET /ready` on the index and report `503` until it's ready. nginx's healthcheck on the APIs ensures no traffic reaches them before the index is up.

## Scoring

The challenge scores on two axes:

```
score_p99  = 1000 × log10(1000 / p99_ms)   # max when p99 = 1ms
score_det  = 1000 × log10(1/ε) − 300 × log10(1 + E)
             where E = FP×1 + FN×3 + Errors×5
```

Failure rate > 15% triggers a −3000 penalty.

Fallback strategy: on index timeout/error, return `approved:false, score:1.0` (assume fraud). This incurs a FP penalty (weight 1) rather than an HTTP error (weight 5) or a missed fraud (weight 3).

## Running locally

```bash
docker compose up -d
```

The index container takes ~2–3 seconds to load. Poll `/ready` to know when everything is up:

```bash
until curl -sf http://localhost:9999/ready; do sleep 1; done
```

Example request:

```bash
curl -s -X POST http://localhost:9999/fraud-score \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "tx-001",
    "transaction": {"amount": 150.0, "installments": 1, "requested_at": "2025-01-15T14:30:00Z"},
    "customer": {"avg_amount": 120.0, "tx_count_24h": 2, "known_merchants": ["merchant-123"]},
    "merchant": {"id": "merchant-456", "mcc": "5812", "avg_amount": 80.0},
    "terminal": {"is_online": true, "card_present": false, "km_from_home": 3.5},
    "last_transaction": {"timestamp": "2025-01-15T12:00:00Z", "km_from_current": 1.2}
  }'
```

## Load test

```bash
k6 run loadtest.js
```

Target: p99 < 2000ms at 900 req/sec sustained.
