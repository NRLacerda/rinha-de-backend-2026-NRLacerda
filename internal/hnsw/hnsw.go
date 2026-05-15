package hnsw

import (
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
)

const Dims = 14
const maxEf = 512

// HNSW is a Hierarchical Navigable Small World graph index.
//
// Memory layout (for N=3M, M=8, int8-quantized vectors):
//   vectors   [N*14]uint8    =  42 MB  (quantized float32→uint8, 4× smaller)
//   conn0     [N*M0]int32    = 192 MB  (M0=16 at M=8)
//   conn0cnt  [N]uint8       =   3 MB
//   visitMark [N]uint32      =  12 MB
//   upperConns map ~sparse   =  ~30 MB
//   qScale/qZero [14]float32 =  <1 MB
//   Go runtime               =  ~15 MB
//   Total                    = ~294 MB  (fits in 316 MB)
type HNSW struct {
	M, M0, EfConstruction int
	mL                    float64

	N       int
	ep      int32
	epLevel int

	// Per-dimension quantization params: stored_val = (float32_val - qZero) / qScale
	// Dequantize: float32_val = float32(stored_val)*qScale + qZero
	qScale [Dims]float32
	qZero  [Dims]float32

	vectors   []uint8 // [N * Dims] quantized as uint8
	nodeLevel []uint8 // [N] max level — nil after Load

	conn0    []int32 // [N * M0]
	conn0cnt []uint8 // [N]

	upperConns map[int32][][]int32

	visitGen  atomic.Uint32
	visitMark []uint32 // [N]
}

// New creates an empty HNSW. Call SetQuantization before AddVector.
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
	// Default: identity quantization (scale=1, zero=0) so uint8 values map directly.
	for d := range h.qScale {
		h.qScale[d] = 1
	}
	for i := range h.conn0 {
		h.conn0[i] = -1
	}
	return h
}

// SetQuantization configures per-dimension dequantization params.
// Must be called before AddVector. scale[d] and zero[d] satisfy:
//
//	float32_value ≈ float32(stored_uint8) * scale[d] + zero[d]
func (h *HNSW) SetQuantization(scale, zero [Dims]float32) {
	h.qScale = scale
	h.qZero = zero
}

// AddVector quantizes vec to uint8 and stores it at slot id.
func (h *HNSW) AddVector(id int32, vec []float32) {
	base := int(id) * Dims
	for d := 0; d < Dims; d++ {
		v := (vec[d] - h.qZero[d]) / h.qScale[d]
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		h.vectors[base+d] = uint8(v + 0.5) // round
	}
}

// ---- Distance ---------------------------------------------------------------

// dist computes Euclidean distance from float32 query q to quantized stored node id.
// Unrolled for 14 dims.
func (h *HNSW) dist(q []float32, id int32) float32 {
	v := h.vectors[int(id)*Dims : int(id)*Dims+Dims]
	d0 := q[0] - (float32(v[0])*h.qScale[0] + h.qZero[0])
	d1 := q[1] - (float32(v[1])*h.qScale[1] + h.qZero[1])
	d2 := q[2] - (float32(v[2])*h.qScale[2] + h.qZero[2])
	d3 := q[3] - (float32(v[3])*h.qScale[3] + h.qZero[3])
	d4 := q[4] - (float32(v[4])*h.qScale[4] + h.qZero[4])
	d5 := q[5] - (float32(v[5])*h.qScale[5] + h.qZero[5])
	d6 := q[6] - (float32(v[6])*h.qScale[6] + h.qZero[6])
	d7 := q[7] - (float32(v[7])*h.qScale[7] + h.qZero[7])
	d8 := q[8] - (float32(v[8])*h.qScale[8] + h.qZero[8])
	d9 := q[9] - (float32(v[9])*h.qScale[9] + h.qZero[9])
	d10 := q[10] - (float32(v[10])*h.qScale[10] + h.qZero[10])
	d11 := q[11] - (float32(v[11])*h.qScale[11] + h.qZero[11])
	d12 := q[12] - (float32(v[12])*h.qScale[12] + h.qZero[12])
	d13 := q[13] - (float32(v[13])*h.qScale[13] + h.qZero[13])
	return float32(math.Sqrt(float64(
		d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 +
			d5*d5 + d6*d6 + d7*d7 + d8*d8 + d9*d9 +
			d10*d10 + d11*d11 + d12*d12 + d13*d13,
	)))
}

// distStored computes Euclidean distance between two stored (quantized) nodes.
// Used during build-time neighbor pruning.
// Since zero cancels in subtraction: (a-zero)*scale - (b-zero)*scale = (a-b)*scale
func (h *HNSW) distStored(a, b []uint8) float32 {
	d0 := (float32(a[0]) - float32(b[0])) * h.qScale[0]
	d1 := (float32(a[1]) - float32(b[1])) * h.qScale[1]
	d2 := (float32(a[2]) - float32(b[2])) * h.qScale[2]
	d3 := (float32(a[3]) - float32(b[3])) * h.qScale[3]
	d4 := (float32(a[4]) - float32(b[4])) * h.qScale[4]
	d5 := (float32(a[5]) - float32(b[5])) * h.qScale[5]
	d6 := (float32(a[6]) - float32(b[6])) * h.qScale[6]
	d7 := (float32(a[7]) - float32(b[7])) * h.qScale[7]
	d8 := (float32(a[8]) - float32(b[8])) * h.qScale[8]
	d9 := (float32(a[9]) - float32(b[9])) * h.qScale[9]
	d10 := (float32(a[10]) - float32(b[10])) * h.qScale[10]
	d11 := (float32(a[11]) - float32(b[11])) * h.qScale[11]
	d12 := (float32(a[12]) - float32(b[12])) * h.qScale[12]
	d13 := (float32(a[13]) - float32(b[13])) * h.qScale[13]
	return float32(math.Sqrt(float64(
		d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 +
			d5*d5 + d6*d6 + d7*d7 + d8*d8 + d9*d9 +
			d10*d10 + d11*d11 + d12*d12 + d13*d13,
	)))
}

// ---- pair type and zero-alloc heaps ----------------------------------------

type pair struct {
	d  float32
	id int32
}

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

func (h *HNSW) searchLayer(q []float32, ep int32, ef, level int, qs *queryState) {
	gen := h.visitGen.Add(1)
	h.visitMark[ep] = gen

	d0 := h.dist(q, ep)
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
			if nb < 0 || h.visitMark[nb] == gen {
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
	// Dequantize the stored vector to use as float32 search query.
	var qArr [Dims]float32
	base := int(id) * Dims
	for d := 0; d < Dims; d++ {
		qArr[d] = float32(h.vectors[base+d])*h.qScale[d] + h.qZero[d]
	}
	q := qArr[:]

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

	qs := &queryState{}
	qs.cands = qs.candsArr[:0]
	qs.results = qs.resultsArr[:0]

	for l := curLevel; l > lvl; l-- {
		h.searchLayer(q, ep, 1, l, qs)
		ep = closestInResults(qs.results)
	}

	for l := min(curLevel, lvl); l >= 0; l-- {
		mMax := h.M
		if l == 0 {
			mMax = h.M0
		}

		h.searchLayer(q, ep, h.EfConstruction, l, qs)

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
					pairs[i] = pair{h.distStored(nbVec, h.vectors[int(n)*Dims:int(n)*Dims+Dims]), n}
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

	qs := statePool.Get().(*queryState)
	qs.cands = qs.candsArr[:0]
	qs.results = qs.resultsArr[:0]
	defer statePool.Put(qs)

	qSlice := q[:]
	ep := h.ep

	for l := h.epLevel; l > 0; l-- {
		h.searchLayer(qSlice, ep, 1, l, qs)
		ep = qs.results[0].id
	}
	h.searchLayer(qSlice, ep, ef, 0, qs)

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

// Vectors returns a dequantized float32 copy of all stored vectors.
// Allocates N*Dims float32s — intended for testing only.
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
