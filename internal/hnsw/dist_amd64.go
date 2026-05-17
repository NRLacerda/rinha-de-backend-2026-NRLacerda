//go:build amd64

package hnsw

//go:noescape
func distSSE2(q *int16, v *uint8) uint32

func (h *HNSW) dist(q []int16, id int32) uint32 {
	return distSSE2(&q[0], &h.vectors[int(id)*stride])
}
