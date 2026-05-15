package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/NRLacerda/rinha-de-backend-2026/internal/hnsw"
)

const (
	M   = 8   // HNSW connections per node (M0=16 at level 0)
	efC = 200 // efConstruction — higher = better recall, slower build
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
	if _, err := dec.Token(); err != nil { // consume '['
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

	// Second pass: load into arrays.
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

	log.Printf("building HNSW (M=%d, efConstruction=%d)...", M, efC)
	t1 := time.Now()

	idx := hnsw.New(n, M, efC)
	for i := 0; i < n; i++ {
		idx.AddVector(int32(i), vectors[i*hnsw.Dims:i*hnsw.Dims+hnsw.Dims])
	}
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
