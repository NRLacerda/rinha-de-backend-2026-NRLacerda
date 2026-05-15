package hnsw

import (
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
)

const Dims = 14
const maxEf = 512 // max ef we'll ever use; bounds the pooled arrays

// HNSW is a Hierarchical Navigable Small World graph index.
//
// Memory layout (for N=3M, M=8):
//   vectors   [N*14]float32  = 168 MB
//   conn0     [N*M0]int32    =  96 MB  (M0=2*M=16 for M=8, or adjust M)
//   conn0cnt  [N]uint8       =   3 MB
//   nodeLevel [N]uint8       =   3 MB
//   visitMark [N]uint64      =  24 MB  (generation-based visited tracking)
//   upper     ~sparse        =  ~15 MB
//   Total                    = ~309 MB  (fits in 316 MB service budget at M=8)
type HNSW struct {
	M, M0, EfConstruction int
	mL                    float64 // 1/ln(M)

	N       int
	ep      int32 // entry point node id
	epLevel int   // level of entry point

	vectors   []float32 // [N * Dims]
	nodeLevel []uint8   // [N] max level per node

	// Level-0 connections (all nodes): fast flat access
	conn0    []int32 // [N * M0]
	conn0cnt []uint8 // [N] actual count

	// Upper-level connections (sparse: only nodes with nodeLevel > 0).
	// upperConns[nodeID] = nil (level-0-only) or [][]int32 indexed by level-1..
	upperConns [][][]int32 // [N][][]int32 — outer slice is large but inner mostly nil

	// Generation-based visited set — safe for concurrent queries.
	visitGen  atomic.Uint64
	visitMark []uint64 // [N]
}

// New creates an empty HNSW ready for insertions.
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
		vectors:        make([]float32, n*Dims),
		nodeLevel:      make([]uint8, n),
		conn0:          make([]int32, n*m0),
		conn0cnt:       make([]uint8, n),
		upperConns:     make([][][]int32, n),
		visitMark:      make([]uint64, n),
	}
	// Init conn0 slots to -1 (empty)
	for i := range h.conn0 {
		h.conn0[i] = -1
	}
	return h
}

// ---- Distance ---------------------------------------------------------------

// dist computes Euclidean distance from query vector q to stored node id.
// Unrolled for 14 dims — helps compiler emit SIMD on amd64.
func (h *HNSW) dist(q []float32, id int32) float32 {
	v := h.vectors[int(id)*Dims : int(id)*Dims+Dims]
	d0 := q[0] - v[0]
	d1 := q[1] - v[1]
	d2 := q[2] - v[2]
	d3 := q[3] - v[3]
	d4 := q[4] - v[4]
	d5 := q[5] - v[5]
	d6 := q[6] - v[6]
	d7 := q[7] - v[7]
	d8 := q[8] - v[8]
	d9 := q[9] - v[9]
	d10 := q[10] - v[10]
	d11 := q[11] - v[11]
	d12 := q[12] - v[12]
	d13 := q[13] - v[13]
	return float32(math.Sqrt(float64(
		d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 +
			d5*d5 + d6*d6 + d7*d7 + d8*d8 + d9*d9 +
			d10*d10 + d11*d11 + d12*d12 + d13*d13,
	)))
}

// ---- pair type --------------------------------------------------------------

type pair struct {
	d  float32
	id int32
}

// ---- Zero-alloc heap (inline, no interface boxing) --------------------------

// queryState holds pre-allocated heaps for a single search call.
// Obtained from statePool; never escapes to the heap.
type queryState struct {
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

// minPush pushes p onto the min-heap h (min distance at root).
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

// minPop removes and returns the minimum element from the min-heap.
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
		smallest := i
		l, r := i*2+1, i*2+2
		if l < n && h[l].d < h[smallest].d {
			smallest = l
		}
		if r < n && h[r].d < h[smallest].d {
			smallest = r
		}
		if smallest == i {
			break
		}
		h[i], h[smallest] = h[smallest], h[i]
		i = smallest
	}
}

// maxPush pushes p onto the max-heap h (max distance at root).
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

// maxPop removes and returns the maximum element from the max-heap.
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
		largest := i
		l, r := i*2+1, i*2+2
		if l < n && h[l].d > h[largest].d {
			largest = l
		}
		if r < n && h[r].d > h[largest].d {
			largest = r
		}
		if largest == i {
			break
		}
		h[i], h[largest] = h[largest], h[i]
		i = largest
	}
}

// minHeapify restores min-heap property on a slice (used in selectNeighbors).
func minHeapify(h []pair) {
	n := len(h)
	for i := n/2 - 1; i >= 0; i-- {
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
		copy(h.conn0[base:base+h.M0], make([]int32, h.M0)) // clear
		for i := range h.conn0[base : base+h.M0] {
			h.conn0[base+i] = -1
		}
		n := copy(h.conn0[base:base+h.M0], nbs)
		h.conn0cnt[id] = uint8(n)
		return
	}
	if h.upperConns[id] == nil {
		h.upperConns[id] = make([][]int32, level)
	}
	for len(h.upperConns[id]) < level {
		h.upperConns[id] = append(h.upperConns[id], nil)
	}
	cp := make([]int32, len(nbs))
	copy(cp, nbs)
	h.upperConns[id][level-1] = cp
}

// ---- Core search ------------------------------------------------------------

// searchLayer performs beam search at the given level, writing results into qs.results.
// On return, qs.results is a max-heap of up to ef nearest candidates found.
func (h *HNSW) searchLayer(q []float32, ep int32, ef, level int, qs *queryState) {
	gen := h.visitGen.Add(1)
	h.visitMark[ep] = gen

	d0 := h.dist(q, ep)

	qs.cands = qs.cands[:0]
	qs.results = qs.results[:0]

	minPush(&qs.cands, pair{d: d0, id: ep})
	maxPush(&qs.results, pair{d: d0, id: ep})

	for len(qs.cands) > 0 {
		c := minPop(&qs.cands)
		fDist := qs.results[0].d // max-heap root = furthest
		if c.d > fDist && len(qs.results) >= ef {
			break
		}
		for _, nb := range h.neighborsAt(c.id, level) {
			if nb < 0 {
				continue
			}
			if h.visitMark[nb] == gen {
				continue
			}
			h.visitMark[nb] = gen

			nd := h.dist(q, nb)
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

// selectNeighbors returns up to M closest candidates from W (simple heuristic).
// Operates in-place: heapifies W then pops m elements into a new slice.
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

// AddVector stores a vector without inserting into the graph (used during bulk load).
func (h *HNSW) AddVector(id int32, vec []float32) {
	copy(h.vectors[int(id)*Dims:int(id)*Dims+Dims], vec)
}

// Insert inserts node id into the graph. Vector must already be stored via AddVector.
func (h *HNSW) Insert(id int32) {
	q := h.vectors[int(id)*Dims : int(id)*Dims+Dims]

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

	// Use a dedicated queryState for insertion (not from the query pool).
	qs := &queryState{}
	qs.cands = qs.candsArr[:0]
	qs.results = qs.resultsArr[:0]

	// Phase 1: greedy descent from curLevel down to lvl+1 (ef=1 per level).
	for l := curLevel; l > lvl; l-- {
		h.searchLayer(q, ep, 1, l, qs)
		ep = closestInResults(qs.results)
	}

	// Phase 2: search + bidirectional connect from min(curLevel,lvl) down to 0.
	for l := min(curLevel, lvl); l >= 0; l-- {
		mMax := h.M
		if l == 0 {
			mMax = h.M0
		}

		h.searchLayer(q, ep, h.EfConstruction, l, qs)

		// Copy results out (selectNeighbors may modify in-place).
		W := make([]pair, len(qs.results))
		copy(W, qs.results)

		neighbors := selectNeighbors(W, mMax)

		// Connect new node → neighbors.
		ids := make([]int32, len(neighbors))
		for i, nb := range neighbors {
			ids[i] = nb.id
		}
		h.setNeighbors(id, ids, l)

		// Connect each neighbor → new node (correct bidirectional wiring with pruning).
		for _, nb := range neighbors {
			cur := h.neighborsAt(nb.id, l)
			combined := make([]int32, len(cur)+1)
			copy(combined, cur)
			combined[len(cur)] = id

			if len(combined) <= mMax {
				h.setNeighbors(nb.id, combined, l)
			} else {
				// Prune: pick best mMax from (current neighbors + new node).
				nbVec := h.vectors[int(nb.id)*Dims : int(nb.id)*Dims+Dims]
				pairs := make([]pair, len(combined))
				for i, n := range combined {
					pairs[i] = pair{distVec(nbVec, h.vectors[int(n)*Dims:int(n)*Dims+Dims]), n}
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

// closestInResults scans a max-heap result set for the element with minimum distance.
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

// distVec is a standalone distance function used during build-time pruning.
func distVec(a, b []float32) float32 {
	d0 := a[0] - b[0]
	d1 := a[1] - b[1]
	d2 := a[2] - b[2]
	d3 := a[3] - b[3]
	d4 := a[4] - b[4]
	d5 := a[5] - b[5]
	d6 := a[6] - b[6]
	d7 := a[7] - b[7]
	d8 := a[8] - b[8]
	d9 := a[9] - b[9]
	d10 := a[10] - b[10]
	d11 := a[11] - b[11]
	d12 := a[12] - b[12]
	d13 := a[13] - b[13]
	return float32(math.Sqrt(float64(
		d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 +
			d5*d5 + d6*d6 + d7*d7 + d8*d8 + d9*d9 +
			d10*d10 + d11*d11 + d12*d12 + d13*d13,
	)))
}

// ---- Query ------------------------------------------------------------------

// Query returns fraud_score ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}.
// labels[i] = 1 if record i is fraud, 0 if legit.
func (h *HNSW) Query(q [Dims]float32, labels []uint8, ef int) float32 {
	if h.ep < 0 {
		return 0
	}

	qs := statePool.Get().(*queryState)
	qs.cands = qs.candsArr[:0]
	qs.results = qs.resultsArr[:0]
	defer statePool.Put(qs)

	qSlice := q[:]
	ep := h.ep

	// Greedy descent to level 1.
	for l := h.epLevel; l > 0; l-- {
		h.searchLayer(qSlice, ep, 1, l, qs)
		ep = qs.results[0].id // ef=1 so only one element; it's both min and max
	}

	// Beam search at level 0 with ef candidates.
	h.searchLayer(qSlice, ep, ef, 0, qs)

	// Take k=5 nearest from max-heap results.
	const k = 5
	results := qs.results
	minHeapify(results)
	var frauds float32
	for i := 0; i < k && len(results) > 0; i++ {
		p := minPop(&results)
		frauds += float32(labels[p.id])
	}
	return frauds / float32(k)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (h *HNSW) Ep() int32    { return h.ep }
func (h *HNSW) EpLevel() int { return h.epLevel }
