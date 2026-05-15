package vptree

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
)

const (
	dims        = 14
	k           = 5
	null        = int32(-1)
	maxLeafSize = 64 // small leaves → heap fills fast → tau shrinks → aggressive pruning
)

// nodeEntry represents one internal node.
// Leaves have left == null; the leaf data lives in Tree.leafVecs/leafLabels.
// VP data points are included in the leaf partition (not stored separately in internal nodes),
// so internal nodes are purely structural — no scoring, only pruning.
type nodeEntry struct {
	vp     [dims]float32
	radius float32
	left   int32
	right  int32
	leafOff int32
	leafLen int32
}

// Tree is the bucketed VP-Tree.
//   - nodes: internal nodes — tiny, stays in cache
//   - leafVecs / leafLabels: contiguous per bucket, brute-forced at query time
type Tree struct {
	nodes      []nodeEntry
	leafVecs   []float32
	leafLabels []uint8
	root       int32
	n          int
}

// Record is used only during Build.
type Record struct {
	Vector [dims]float32
	Label  uint8
}

// ---- Distance ---------------------------------------------------------------

// dist returns Euclidean distance. Called ~log(N/leaf) times per query (internal nodes only).
func dist(a, b []float32) float32 {
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

// distSq returns squared Euclidean distance (no sqrt). Called ~maxLeafSize times per query.
func distSq(a, b []float32) float32 {
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
	return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 +
		d5*d5 + d6*d6 + d7*d7 + d8*d8 + d9*d9 +
		d10*d10 + d11*d11 + d12*d12 + d13*d13
}

// ---- Build ------------------------------------------------------------------

func Build(records []Record) *Tree {
	n := len(records)
	t := &Tree{
		nodes:      make([]nodeEntry, 0, 2*n/maxLeafSize+16),
		leafVecs:   make([]float32, 0, n*dims),
		leafLabels: make([]uint8, 0, n),
		n:          n,
	}
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	t.root = t.buildNode(records, indices)
	return t
}

func (t *Tree) buildNode(records []Record, indices []int) int32 {
	slot := int32(len(t.nodes))
	t.nodes = append(t.nodes, nodeEntry{})

	if len(indices) <= maxLeafSize {
		leafOff := int32(len(t.leafLabels))
		for _, idx := range indices {
			t.leafVecs = append(t.leafVecs, records[idx].Vector[:]...)
			t.leafLabels = append(t.leafLabels, records[idx].Label)
		}
		t.nodes[slot] = nodeEntry{
			left:    null,
			right:   null,
			leafOff: leafOff,
			leafLen: int32(len(indices)),
		}
		return slot
	}

	// Pick vantage point (structural only — stays in indices, included in a leaf).
	vpPos := selectVantagePoint(records, indices)
	vp := records[indices[vpPos]]

	// Compute distances from VP to all points (including VP itself at dist=0).
	dists := make([]float32, len(indices))
	for i, idx := range indices {
		dists[i] = dist(vp.Vector[:], records[idx].Vector[:])
	}

	median := medianF32(dists)

	// Partition all points (including VP) into inner/outer by median distance.
	inner := make([]int, 0, len(indices)/2)
	outer := make([]int, 0, len(indices)/2)
	for i, idx := range indices {
		if dists[i] <= median {
			inner = append(inner, idx)
		} else {
			outer = append(outer, idx)
		}
	}
	// Guarantee neither side is empty (can happen if all dists == median).
	if len(inner) == 0 {
		inner, outer = outer[:1], outer[1:]
	} else if len(outer) == 0 {
		outer = inner[len(inner)-1:]
		inner = inner[:len(inner)-1]
	}

	leftChild := t.buildNode(records, inner)
	rightChild := t.buildNode(records, outer)

	t.nodes[slot] = nodeEntry{
		vp:      vp.Vector,
		radius:  median,
		left:    leftChild,
		right:   rightChild,
		leafOff: -1,
		leafLen: 0,
	}
	return slot
}

const (
	sampleCandidates = 20
	sampleProbes     = 20
)

func selectVantagePoint(records []Record, indices []int) int {
	if len(indices) <= sampleCandidates {
		return rand.Intn(len(indices))
	}
	candidates := rand.Perm(len(indices))[:sampleCandidates]
	probes := rand.Perm(len(indices))[:sampleProbes]

	bestIdx, bestVar := 0, float32(-1)
	n := float32(sampleProbes)
	for _, ci := range candidates {
		vp := records[indices[ci]].Vector[:]
		var sum, sum2 float32
		for _, pi := range probes {
			d := dist(vp, records[indices[pi]].Vector[:])
			sum += d
			sum2 += d * d
		}
		v := sum2/n - (sum/n)*(sum/n)
		if v > bestVar {
			bestVar = v
			bestIdx = ci
		}
	}
	return bestIdx
}

func medianF32(a []float32) float32 {
	quickselect(a, len(a)/2)
	return a[len(a)/2]
}

func quickselect(a []float32, target int) {
	lo, hi := 0, len(a)-1
	for lo < hi {
		p := partitionF32(a, lo, hi)
		if p == target {
			return
		} else if p < target {
			lo = p + 1
		} else {
			hi = p - 1
		}
	}
}

func partitionF32(a []float32, lo, hi int) int {
	pivot := a[hi]
	i := lo
	for j := lo; j < hi; j++ {
		if a[j] <= pivot {
			a[i], a[j] = a[j], a[i]
			i++
		}
	}
	a[i], a[hi] = a[hi], a[i]
	return i
}

// ---- Query ------------------------------------------------------------------

// heap5: fixed max-heap of k=5 (squared-distance, label) pairs.
// Leaf comparisons use squared distances (no sqrt).
// Pruning at internal nodes uses actual distance (sqrt once per internal node).
type heap5 struct {
	dists  [k]float32 // squared distances
	labels [k]uint8
	size   int
}

func (h *heap5) worstSq() float32 {
	if h.size < k {
		return math.MaxFloat32
	}
	return h.dists[0]
}

func (h *heap5) worstDist() float32 {
	return float32(math.Sqrt(float64(h.worstSq())))
}

func (h *heap5) push(dsq float32, label uint8) {
	if h.size < k {
		h.dists[h.size] = dsq
		h.labels[h.size] = label
		h.size++
		if h.size == k {
			buildHeap5(h)
		}
		return
	}
	if dsq < h.dists[0] {
		h.dists[0] = dsq
		h.labels[0] = label
		siftDown5(h, 0)
	}
}

func buildHeap5(h *heap5) {
	for i := k/2 - 1; i >= 0; i-- {
		siftDown5(h, i)
	}
}

func siftDown5(h *heap5, i int) {
	for {
		largest := i
		l, r := 2*i+1, 2*i+2
		if l < h.size && h.dists[l] > h.dists[largest] {
			largest = l
		}
		if r < h.size && h.dists[r] > h.dists[largest] {
			largest = r
		}
		if largest == i {
			break
		}
		h.dists[i], h.dists[largest] = h.dists[largest], h.dists[i]
		h.labels[i], h.labels[largest] = h.labels[largest], h.labels[i]
		i = largest
	}
}

// Query returns fraud_score ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}.
func (t *Tree) Query(q [dims]float32) float32 {
	var h heap5
	t.search(q[:], t.root, &h)
	var frauds float32
	for i := 0; i < h.size; i++ {
		frauds += float32(h.labels[i])
	}
	return frauds / float32(k)
}

func (t *Tree) search(q []float32, nodeIdx int32, h *heap5) {
	if nodeIdx == null {
		return
	}
	nd := &t.nodes[nodeIdx]

	if nd.left == null {
		// Leaf: brute-force sequential scan — cache-friendly, no sqrt.
		off := nd.leafOff
		end := off + nd.leafLen
		for i := off; i < end; i++ {
			v := t.leafVecs[i*dims : i*dims+dims]
			h.push(distSq(q, v), t.leafLabels[i])
		}
		return
	}

	// Internal node: pure structural — use VP only for pruning, not scoring.
	// VP data points are included in leaf partitions.
	d := dist(q, nd.vp[:])

	tau := h.worstDist()
	radius := nd.radius

	if d <= radius {
		t.search(q, nd.left, h)
		tau = h.worstDist()
		if d+tau >= radius {
			t.search(q, nd.right, h)
		}
	} else {
		t.search(q, nd.right, h)
		tau = h.worstDist()
		if d-tau <= radius {
			t.search(q, nd.left, h)
		}
	}
}

// ---- Persistence ------------------------------------------------------------

// Binary format (little-endian):
//   int32  n_nodes
//   int32  n_leaf_vecs
//   []nodeEntry  (fixed-size struct, written field by field)
//   []float32    leafVecs
//   []uint8      leafLabels

func (t *Tree) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	nNodes := int32(len(t.nodes))
	nLeaf := int32(len(t.leafLabels))
	if err := binary.Write(f, binary.LittleEndian, nNodes); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, nLeaf); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, t.root); err != nil {
		return err
	}

	for i := range t.nodes {
		nd := &t.nodes[i]
		if err := binary.Write(f, binary.LittleEndian, nd.vp); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, nd.radius); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, nd.left); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, nd.right); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, nd.leafOff); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, nd.leafLen); err != nil {
			return err
		}
	}

	if err := binary.Write(f, binary.LittleEndian, t.leafVecs); err != nil {
		return err
	}
	_, err = f.Write(t.leafLabels)
	return err
}

func Load(path string) (*Tree, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	var nNodes, nLeaf, root int32
	if err := binary.Read(f, binary.LittleEndian, &nNodes); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &nLeaf); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &root); err != nil {
		return nil, err
	}

	nodes := make([]nodeEntry, nNodes)
	for i := range nodes {
		nd := &nodes[i]
		if err := binary.Read(f, binary.LittleEndian, &nd.vp); err != nil {
			return nil, err
		}
		if err := binary.Read(f, binary.LittleEndian, &nd.radius); err != nil {
			return nil, err
		}
		if err := binary.Read(f, binary.LittleEndian, &nd.left); err != nil {
			return nil, err
		}
		if err := binary.Read(f, binary.LittleEndian, &nd.right); err != nil {
			return nil, err
		}
		if err := binary.Read(f, binary.LittleEndian, &nd.leafOff); err != nil {
			return nil, err
		}
		if err := binary.Read(f, binary.LittleEndian, &nd.leafLen); err != nil {
			return nil, err
		}
	}

	leafVecs := make([]float32, int(nLeaf)*dims)
	if err := binary.Read(f, binary.LittleEndian, leafVecs); err != nil {
		return nil, err
	}
	leafLabels := make([]uint8, nLeaf)
	if _, err := f.Read(leafLabels); err != nil {
		return nil, err
	}

	return &Tree{
		nodes:      nodes,
		leafVecs:   leafVecs,
		leafLabels: leafLabels,
		root:       root,
		n:          int(nLeaf),
	}, nil
}
