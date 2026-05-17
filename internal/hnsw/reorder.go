package hnsw

// Reorder remaps all node IDs to BFS order starting from the entry point on
// layer 0. Nodes visited earlier during a query will have small IDs, keeping
// their vectors and conn0 entries in nearby memory — better cache locality.
// Returns the labels slice reordered to match the new IDs.
// Call this after building the index, before saving.
func (h *HNSW) Reorder(labels []uint8) []uint8 {
	n := h.N

	newID := make([]int32, n)
	for i := range newID {
		newID[i] = -1
	}
	oldByNew := make([]int32, 0, n)

	enqueue := func(old int32) {
		if old >= 0 && newID[old] == -1 {
			newID[old] = int32(len(oldByNew))
			oldByNew = append(oldByNew, old)
		}
	}

	enqueue(h.ep)
	for qi := 0; qi < len(oldByNew); qi++ {
		cur := oldByNew[qi]
		base := int(cur) * h.M0
		cnt := int(h.conn0cnt[cur])
		for _, nb := range h.conn0[base : base+cnt] {
			enqueue(nb)
		}
	}
	// Safety: append any nodes unreachable from ep.
	for i := 0; i < n; i++ {
		enqueue(int32(i))
	}

	newVec := make([]uint8, n*stride)
	for newI, oldI := range oldByNew {
		copy(newVec[newI*Dims:], h.vectors[int(oldI)*stride:int(oldI)*stride+stride])
	}
	h.vectors = newVec

	newLabels := make([]uint8, n)
	for newI, oldI := range oldByNew {
		newLabels[newI] = labels[oldI]
	}

	newConn0 := make([]int32, n*h.M0)
	newConn0cnt := make([]uint8, n)
	for i := range newConn0 {
		newConn0[i] = -1
	}
	for newI, oldI := range oldByNew {
		base := int(oldI) * h.M0
		cnt := int(h.conn0cnt[oldI])
		newBase := newI * h.M0
		newConn0cnt[newI] = uint8(cnt)
		for j := 0; j < cnt; j++ {
			if nb := h.conn0[base+j]; nb >= 0 {
				newConn0[newBase+j] = newID[nb]
			}
		}
	}
	h.conn0 = newConn0
	h.conn0cnt = newConn0cnt

	newUpper := make(map[int32][][]int32, len(h.upperConns))
	for oldI, uc := range h.upperConns {
		newI := newID[oldI]
		newUC := make([][]int32, len(uc))
		for l, lvlNbs := range uc {
			remapped := make([]int32, len(lvlNbs))
			for j, nb := range lvlNbs {
				remapped[j] = newID[nb]
			}
			newUC[l] = remapped
		}
		newUpper[newI] = newUC
	}
	h.upperConns = newUpper

	h.ep = newID[h.ep]

	return newLabels
}
