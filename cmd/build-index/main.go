package main

import (
	"encoding/json"
	"log"
	"math"
	"os"

	"github.com/NRLacerda/rinha-de-backend-2026/internal/hnsw"
)

const (
	M   = 8   // HNSW connections per node (M0=16)
	efC = 200 // efConstruction
)

type jsonRecord struct {
	Vector [hnsw.Dims]float32 `json:"vector"`
	Label  string             `json:"label"`
}

func main() {
	// First pass: count records.
	f, err := os.Open("resources/references.json")
	if err != nil {
		log.Fatalf("open: %v", err)
	}
	var n int
	dec := json.NewDecoder(f)
	if _, err := dec.Token(); err != nil {
		log.Fatalf("read '[': %v", err)
	}
	for dec.More() {
		var r jsonRecord
		if err := dec.Decode(&r); err != nil {
			log.Fatalf("decode: %v", err)
		}
		n++
	}
	f.Close()

	// Second pass: load vectors + labels.
	f, err = os.Open("resources/references.json")
	if err != nil {
		log.Fatalf("open: %v", err)
	}
	dec = json.NewDecoder(f)
	if _, err := dec.Token(); err != nil {
		log.Fatalf("read '[': %v", err)
	}

	vectors := make([]float32, n*hnsw.Dims)
	labels := make([]uint8, n)

	for i := 0; dec.More(); i++ {
		var r jsonRecord
		if err := dec.Decode(&r); err != nil {
			log.Fatalf("decode %d: %v", i, err)
		}
		copy(vectors[i*hnsw.Dims:], r.Vector[:])
		if r.Label == "fraud" {
			labels[i] = 1
		}
	}
	f.Close()

	// Compute per-dimension min/max for uint8 quantization.
	var dimMin, dimMax [hnsw.Dims]float32
	for d := range dimMin {
		dimMin[d] = math.MaxFloat32
		dimMax[d] = -math.MaxFloat32
	}
	for i := 0; i < n; i++ {
		for d := 0; d < hnsw.Dims; d++ {
			v := vectors[i*hnsw.Dims+d]
			if v < dimMin[d] {
				dimMin[d] = v
			}
			if v > dimMax[d] {
				dimMax[d] = v
			}
		}
	}

	var scale, zero [hnsw.Dims]float32
	for d := 0; d < hnsw.Dims; d++ {
		r := dimMax[d] - dimMin[d]
		if r == 0 {
			r = 1
		}
		scale[d] = r / 255.0
		zero[d] = dimMin[d]
	}

	idx := hnsw.New(n, M, efC)
	idx.SetQuantization(scale, zero)

	for i := 0; i < n; i++ {
		idx.AddVector(int32(i), vectors[i*hnsw.Dims:i*hnsw.Dims+hnsw.Dims])
	}
	vectors = nil

	for i := 0; i < n; i++ {
		idx.Insert(int32(i))
	}

	if err := idx.SaveWithLabels("resources/hnsw.bin", labels); err != nil {
		log.Fatalf("save: %v", err)
	}
}
