//go:build !amd64

package hnsw

func (h *HNSW) dist(q []int16, id int32) uint32 {
	vBase := int(id) * stride
	var sum uint32
	for d := 0; d < stride; d++ {
		diff := int32(q[d]) - int32(h.vectors[vBase+d])
		sum += uint32(diff * diff)
	}
	return sum
}
