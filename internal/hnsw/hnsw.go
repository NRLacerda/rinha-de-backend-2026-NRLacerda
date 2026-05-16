package hnsw

import (
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
)

// visitSlot holds a per-query visited marker array and its generation counter.
// Concurrent queries each own one slot from HNSW.visitPool, eliminating data races
// on the shared visitMark that existed with the previous single-array design.
type visitSlot struct {
	mark []uint32
	gen  uint32
}

const Dims = 14
const maxEf = 512

// HNSW is a Hierarchical Navigable Small World graph index with uint8-quantized vectors.
//
// Distance metric: squared integer Euclidean in uint8 space (no sqrt, no scale).
// All comparisons use squared distances — ranking is identical to true Euclidean.
// The query float32 vector is quantized to uint8 ONCE per Query call; every
// individual distance computation is then pure integer arithmetic.
//
// Memory layout (N=3M, M=8):
//   vectors   [N*14]uint8    =  42 MB
//   conn0     [N*M0]int32    = 192 MB  (M0=16)
//   conn0cnt  [N]uint8       =   3 MB
//   visitMark [N]uint32      =  12 MB
//   upperConns map ~sparse   =  ~30 MB
//   misc/runtime             =  ~15 MB
//   Total                    = ~294 MB  (fits in 316 MB)
type HNSW struct {
	M, M0, EfConstruction int
	mL                    float64

	N       int
	ep      int32
	epLevel int

	// Quantization params: float32_value ≈ float32(stored_uint8)*qScale[d] + qZero[d]
	// Used only to quantize incoming float32 query vectors — NOT in the distance hot path.
	qScale [Dims]float32
	qZero  [Dims]float32

	vectors   []uint8 // [N * Dims] quantized as uint8
	nodeLevel []uint8 // [N] — nil after Load

	conn0    []int32 // [N * M0]
	conn0cnt []uint8 // [N]

	upperConns map[int32][][]int32

	visitGen  atomic.Uint32
	visitMark []uint32       // [N] — used only during Insert (build time); nil after Load
	visitPool chan *visitSlot // bounded pool for concurrent Query calls
}

func New(n, m, efC int) *HNSW {
	m0 := 2 * m
	h := &HNSW{
		M:              m,
		M0:             m0,
		EfConstruction: efC,
		mL:             1.0 / math.Log(float64(m)),
		N:              0,
		ep:             -1,
		epLevel:        -1,
		vectors:        make([]uint8, n*Dims),
		nodeLevel:      make([]uint8, n),
		conn0:          make([]int32, n*m0),
		conn0cnt:       make([]uint8, n),
		upperConns:     make(map[int32][][]int32),
		visitMark:      make([]uint32, n),
	}
	h.initVisitPool(2, n)
	for d := range h.qScale {
		h.qScale[d] = 1
	}
	for i := range h.conn0 {
		h.conn0[i] = -1
	}
	return h
}

// initVisitPool allocates n visitSlots for concurrent Query use.
// Must be called after h.N is set. Replaces h.visitMark for query-time use.
func (h *HNSW) initVisitPool(poolSize, markSize int) {
	h.visitPool = make(chan *visitSlot, poolSize)
	for i := 0; i < poolSize; i++ {
		h.visitPool <- &visitSlot{mark: make([]uint32, markSize)}
	}
}

// SetQuantization configures per-dimension params used to quantize incoming float32 queries.
func (h *HNSW) SetQuantization(scale, zero [Dims]float32) {
	h.qScale = scale
	h.qZero = zero
}

// AddVector quantizes vec to uint8 and stores at slot id.
func (h *HNSW) AddVector(id int32, vec []float32) {
	base := int(id) * Dims
	for d := 0; d < Dims; d++ {
		v := (vec[d] - h.qZero[d]) / h.qScale[d]
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		h.vectors[base+d] = uint8(v + 0.5)
	}
}

// ---- Distance (pure integer, no sqrt) ---------------------------------------
//
// dist returns SQUARED integer Euclidean distance between pre-quantized int16
// query q and stored uint8 node id. Stored as float32 for use in pair.d.
// Dropping sqrt is safe: all comparisons only need relative ordering.
//
// Unrolled for 14 dims — helps the compiler emit SIMD integer ops.
func (h *HNSW) dist(q []int16, id int32) float32 {
	v := h.vectors[int(id)*Dims : int(id)*Dims+Dims]
	d0 := int32(q[0]) - int32(v[0])
	d1 := int32(q[1]) - int32(v[1])
	d2 := int32(q[2]) - int32(v[2])
	d3 := int32(q[3]) - int32(v[3])
	d4 := int32(q[4]) - int32(v[4])
	d5 := int32(q[5]) - int32(v[5])
	d6 := int32(q[6]) - int32(v[6])
	d7 := int32(q[7]) - int32(v[7])
	d8 := int32(q[8]) - int32(v[8])
	d9 := int32(q[9]) - int32(v[9])
	d10 := int32(q[10]) - int32(v[10])
	d11 := int32(q[11]) - int32(v[11])
	d12 := int32(q[12]) - int32(v[12])
	d13 := int32(q[13]) - int32(v[13])
	return float32(d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 +
		d5*d5 + d6*d6 + d7*d7 + d8*d8 + d9*d9 +
		d10*d10 + d11*d11 + d12*d12 + d13*d13)
}

// distStored returns squared integer Euclidean distance between two stored uint8 nodes.
// Used only during build-time pruning — same metric as dist, but both inputs are uint8.
func distStored(a, b []uint8) float32 {
	d0 := int32(a[0]) - int32(b[0])
	d1 := int32(a[1]) - int32(b[1])
	d2 := int32(a[2]) - int32(b[2])
	d3 := int32(a[3]) - int32(b[3])
	d4 := int32(a[4]) - int32(b[4])
	d5 := int32(a[5]) - int32(b[5])
	d6 := int32(a[6]) - int32(b[6])
	d7 := int32(a[7]) - int32(b[7])
	d8 := int32(a[8]) - int32(b[8])
	d9 := int32(a[9]) - int32(b[9])
	d10 := int32(a[10]) - int32(b[10])
	d11 := int32(a[11]) - int32(b[11])
	d12 := int32(a[12]) - int32(b[12])
	d13 := int32(a[13]) - int32(b[13])
	return float32(d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 +
		d5*d5 + d6*d6 + d7*d7 + d8*d8 + d9*d9 +
		d10*d10 + d11*d11 + d12*d12 + d13*d13)
}

// quantizeQuery converts an incoming float32 query to int16 (uint8 range, signed
// for subtraction). Called once per Query — amortized over all distance computations.
func (h *HNSW) quantizeQuery(q [Dims]float32) [Dims]int16 {
	var qv [Dims]int16
	for d := 0; d < Dims; d++ {
		v := (q[d] - h.qZero[d]) / h.qScale[d]
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		qv[d] = int16(v + 0.5)
	}
	return qv
}

// ---- pair type and zero-alloc heaps ----------------------------------------

type pair struct {
	d  float32 // squared integer distance — comparable as-is
	id int32
}

// queryState holds pre-allocated heaps and the pre-quantized query vector.
type queryState struct {
	qVec       [Dims]int16
	cands      []pair
	results    []pair
	candsArr   [maxEf]pair
	resultsArr [maxEf]pair
}

var statePool = sync.Pool{New: func() any {
	qs := new(queryState)
	qs.cands = qs.candsArr[:0]
	qs.results = qs.resultsArr[:0]
	return qs
}}

func minPush(h *[]pair, p pair) {
	*h = append(*h, p)
	i := len(*h) - 1
	for i > 0 {
		parent := (i - 1) >> 1
		if (*h)[parent].d <= (*h)[i].d {
			break
		}
		(*h)[i], (*h)[parent] = (*h)[parent], (*h)[i]
		i = parent
	}
}

func minPop(h *[]pair) pair {
	n := len(*h) - 1
	(*h)[0], (*h)[n] = (*h)[n], (*h)[0]
	v := (*h)[n]
	*h = (*h)[:n]
	minSiftDown(*h, 0)
	return v
}

func minSiftDown(h []pair, i int) {
	n := len(h)
	for {
		s := i
		l, r := i*2+1, i*2+2
		if l < n && h[l].d < h[s].d {
			s = l
		}
		if r < n && h[r].d < h[s].d {
			s = r
		}
		if s == i {
			break
		}
		h[i], h[s] = h[s], h[i]
		i = s
	}
}

func maxPush(h *[]pair, p pair) {
	*h = append(*h, p)
	i := len(*h) - 1
	for i > 0 {
		parent := (i - 1) >> 1
		if (*h)[parent].d >= (*h)[i].d {
			break
		}
		(*h)[i], (*h)[parent] = (*h)[parent], (*h)[i]
		i = parent
	}
}

func maxPop(h *[]pair) pair {
	n := len(*h) - 1
	(*h)[0], (*h)[n] = (*h)[n], (*h)[0]
	v := (*h)[n]
	*h = (*h)[:n]
	maxSiftDown(*h, 0)
	return v
}

func maxSiftDown(h []pair, i int) {
	n := len(h)
	for {
		lg := i
		l, r := i*2+1, i*2+2
		if l < n && h[l].d > h[lg].d {
			lg = l
		}
		if r < n && h[r].d > h[lg].d {
			lg = r
		}
		if lg == i {
			break
		}
		h[i], h[lg] = h[lg], h[i]
		i = lg
	}
}

func minHeapify(h []pair) {
	for i := len(h)/2 - 1; i >= 0; i-- {
		minSiftDown(h, i)
	}
}

// ---- Neighbor access --------------------------------------------------------

func (h *HNSW) neighborsAt(id int32, level int) []int32 {
	if level == 0 {
		cnt := int(h.conn0cnt[id])
		base := int(id) * h.M0
		return h.conn0[base : base+cnt]
	}
	uc := h.upperConns[id]
	if uc == nil || level-1 >= len(uc) {
		return nil
	}
	return uc[level-1]
}

func (h *HNSW) setNeighbors(id int32, nbs []int32, level int) {
	if level == 0 {
		base := int(id) * h.M0
		for i := range h.conn0[base : base+h.M0] {
			h.conn0[base+i] = -1
		}
		h.conn0cnt[id] = uint8(copy(h.conn0[base:base+h.M0], nbs))
		return
	}
	uc := h.upperConns[id]
	if uc == nil {
		uc = make([][]int32, level)
		h.upperConns[id] = uc
	}
	for len(uc) < level {
		uc = append(uc, nil)
		h.upperConns[id] = uc
	}
	cp := make([]int32, len(nbs))
	copy(cp, nbs)
	uc[level-1] = cp
}

// ---- Core search ------------------------------------------------------------

// searchLayer uses the pre-quantized qs.qVec for all distance computations.
// vs is the caller-owned visitSlot — its gen is incremented once per call.
func (h *HNSW) searchLayer(ep int32, ef, level int, qs *queryState, vs *visitSlot) {
	vs.gen++
	if vs.gen == 0 {
		clear(vs.mark)
		vs.gen = 1
	}
	gen := vs.gen
	mark := vs.mark

	mark[ep] = gen
	d0 := h.dist(qs.qVec[:], ep)
	qs.cands = qs.cands[:0]
	qs.results = qs.results[:0]
	minPush(&qs.cands, pair{d0, ep})
	maxPush(&qs.results, pair{d0, ep})

	for len(qs.cands) > 0 {
		c := minPop(&qs.cands)
		if c.d > qs.results[0].d && len(qs.results) >= ef {
			break
		}
		for _, nb := range h.neighborsAt(c.id, level) {
			if nb < 0 || mark[nb] == gen {
				continue
			}
			mark[nb] = gen
			nd := h.dist(qs.qVec[:], nb)
			if len(qs.results) < ef || nd < qs.results[0].d {
				minPush(&qs.cands, pair{nd, nb})
				maxPush(&qs.results, pair{nd, nb})
				if len(qs.results) > ef {
					maxPop(&qs.results)
				}
			}
		}
	}
}

func selectNeighbors(candidates []pair, m int) []pair {
	if len(candidates) <= m {
		return candidates
	}
	minHeapify(candidates)
	result := make([]pair, 0, m)
	h := candidates
	for i := 0; i < m && len(h) > 0; i++ {
		result = append(result, minPop(&h))
	}
	return result
}

// ---- Insert -----------------------------------------------------------------

func (h *HNSW) Insert(id int32) {
	lvl := int(math.Floor(-math.Log(rand.Float64()) * h.mL))
	if lvl > 16 {
		lvl = 16
	}
	h.nodeLevel[id] = uint8(lvl)

	if h.ep < 0 {
		h.ep = id
		h.epLevel = lvl
		h.N++
		return
	}

	ep := h.ep
	curLevel := h.epLevel

	// Use stored uint8 values directly as int16 — no dequantization needed.
	qs := &queryState{}
	qs.cands = qs.candsArr[:0]
	qs.results = qs.resultsArr[:0]
	base := int(id) * Dims
	for d := 0; d < Dims; d++ {
		qs.qVec[d] = int16(h.vectors[base+d])
	}

	// Build-time visitSlot backed by the struct's visitMark (single-threaded Insert).
	vs := &visitSlot{mark: h.visitMark, gen: uint32(h.visitGen.Load())}
	defer func() { h.visitGen.Store(vs.gen) }()

	for l := curLevel; l > lvl; l-- {
		h.searchLayer(ep, 1, l, qs, vs)
		ep = closestInResults(qs.results)
	}

	for l := min(curLevel, lvl); l >= 0; l-- {
		mMax := h.M
		if l == 0 {
			mMax = h.M0
		}

		h.searchLayer(ep, h.EfConstruction, l, qs, vs)

		W := make([]pair, len(qs.results))
		copy(W, qs.results)
		neighbors := selectNeighbors(W, mMax)

		ids := make([]int32, len(neighbors))
		for i, nb := range neighbors {
			ids[i] = nb.id
		}
		h.setNeighbors(id, ids, l)

		for _, nb := range neighbors {
			cur := h.neighborsAt(nb.id, l)
			combined := make([]int32, len(cur)+1)
			copy(combined, cur)
			combined[len(cur)] = id

			if len(combined) <= mMax {
				h.setNeighbors(nb.id, combined, l)
			} else {
				nbVec := h.vectors[int(nb.id)*Dims : int(nb.id)*Dims+Dims]
				pairs := make([]pair, len(combined))
				for i, n := range combined {
					pairs[i] = pair{distStored(nbVec, h.vectors[int(n)*Dims:int(n)*Dims+Dims]), n}
				}
				pruned := selectNeighbors(pairs, mMax)
				pruneIDs := make([]int32, len(pruned))
				for i, p := range pruned {
					pruneIDs[i] = p.id
				}
				h.setNeighbors(nb.id, pruneIDs, l)
			}
		}

		ep = closestInResults(qs.results)
	}

	h.N++
	if lvl > h.epLevel {
		h.ep = id
		h.epLevel = lvl
	}
}

func closestInResults(W []pair) int32 {
	if len(W) == 0 {
		return -1
	}
	best := W[0]
	for _, p := range W[1:] {
		if p.d < best.d {
			best = p
		}
	}
	return best.id
}

// ---- Query ------------------------------------------------------------------

func (h *HNSW) Query(q [Dims]float32, labels []uint8, ef int) float32 {
	if h.ep < 0 {
		return 0
	}

	vs := <-h.visitPool
	defer func() { h.visitPool <- vs }()

	qs := statePool.Get().(*queryState)
	qs.cands = qs.candsArr[:0]
	qs.results = qs.resultsArr[:0]
	// Quantize the incoming float32 query ONCE — all distance calls reuse qs.qVec.
	qs.qVec = h.quantizeQuery(q)
	defer statePool.Put(qs)

	ep := h.ep
	for l := h.epLevel; l > 0; l-- {
		h.searchLayer(ep, 1, l, qs, vs)
		ep = qs.results[0].id
	}
	h.searchLayer(ep, ef, 0, qs, vs)

	const k = 5
	results := qs.results
	minHeapify(results)
	var frauds float32
	for i := 0; i < k && len(results) > 0; i++ {
		frauds += float32(labels[minPop(&results).id])
	}
	return frauds / float32(k)
}

// ---- Accessors --------------------------------------------------------------

func (h *HNSW) Ep() int32    { return h.ep }
func (h *HNSW) EpLevel() int { return h.epLevel }

// Vectors returns dequantized float32 copies of all stored vectors. Tests only.
func (h *HNSW) Vectors() []float32 {
	out := make([]float32, h.N*Dims)
	for i := 0; i < h.N; i++ {
		for d := 0; d < Dims; d++ {
			out[i*Dims+d] = float32(h.vectors[i*Dims+d])*h.qScale[d] + h.qZero[d]
		}
	}
	return out
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
