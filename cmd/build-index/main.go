package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"github.com/NRLacerda/rinha-de-backend-2026/internal/hnsw"
)

const (
	M   = 8   // HNSW connections per node (M0=16); int8 quantization frees 126 MB to allow M=8
	efC = 200 // efConstruction
)

type jsonRecord struct {
	Vector [hnsw.Dims]float32 `json:"vector"`
	Label  string             `json:"label"`
}

func main() {
	log.Println("loading references.json...")
	t0 := time.Now()

	f, err := os.Open("resources/references.json")
	if err != nil {
		log.Fatalf("open: %v", err)
	}

	// First pass: count records.
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
	log.Printf("counted %d records in %s", n, time.Since(t0))

	// Second pass: load all vectors + labels into memory.
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
		if i%500_000 == 0 {
			fmt.Printf("  loaded %d/%d\n", i, n)
		}
	}
	f.Close()
	log.Printf("loaded in %s", time.Since(t0))

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

	// scale[d] = (max-min)/255  →  uint8 = (float32 - min) / scale
	// zero[d]  = min
	var scale, zero [hnsw.Dims]float32
	for d := 0; d < hnsw.Dims; d++ {
		r := dimMax[d] - dimMin[d]
		if r == 0 {
			r = 1 // avoid div-by-zero for constant dimensions
		}
		scale[d] = r / 255.0
		zero[d] = dimMin[d]
	}
	log.Println("quantization params computed")

	log.Printf("building HNSW (M=%d, efConstruction=%d)...", M, efC)
	t1 := time.Now()

	idx := hnsw.New(n, M, efC)
	idx.SetQuantization(scale, zero)

	for i := 0; i < n; i++ {
		idx.AddVector(int32(i), vectors[i*hnsw.Dims:i*hnsw.Dims+hnsw.Dims])
	}
	// Free the raw float32 vectors — no longer needed after quantization.
	vectors = nil

	for i := 0; i < n; i++ {
		idx.Insert(int32(i))
		if i%500_000 == 0 && i > 0 {
			elapsed := time.Since(t1)
			eta := time.Duration(float64(elapsed) / float64(i) * float64(n-i))
			fmt.Printf("  inserted %d/%d  ETA %s\n", i, n, eta.Round(time.Second))
		}
	}

	log.Printf("build done in %s", time.Since(t1))

	log.Println("saving hnsw.bin...")
	if err := idx.SaveWithLabels("resources/hnsw.bin", labels); err != nil {
		log.Fatalf("save: %v", err)
	}
	log.Println("done")
}
