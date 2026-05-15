package vptree

import (
	"math/rand"
	"os"
	"testing"
)

func BenchmarkQueryReal(b *testing.B) {
	nullPath := "../../resources/vptree-null.bin"
	fullPath := "../../resources/vptree-full.bin"
	if _, err := os.Stat(nullPath); err != nil {
		b.Skip("vptree bins not found, run cmd/build-index first")
	}

	forest, err := LoadForest(nullPath, fullPath)
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))

	// Benchmark with-last-tx queries (80% of real traffic).
	var q [dims]float32
	for i := range q {
		q[i] = rng.Float32()
	}
	// Ensure non-null last_tx (dims 5 & 6 in [0,1]).
	q[5] = rng.Float32()
	q[6] = rng.Float32()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		forest.Query(q)
	}
}

func BenchmarkQueryRealNullTx(b *testing.B) {
	nullPath := "../../resources/vptree-null.bin"
	fullPath := "../../resources/vptree-full.bin"
	if _, err := os.Stat(nullPath); err != nil {
		b.Skip("vptree bins not found, run cmd/build-index first")
	}

	forest, err := LoadForest(nullPath, fullPath)
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(99))
	var q [dims]float32
	for i := range q {
		q[i] = rng.Float32()
	}
	q[5] = -1
	q[6] = -1

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		forest.Query(q)
	}
}

func TestPruningStats(t *testing.T) {
	nullPath := "../../resources/vptree-null.bin"
	fullPath := "../../resources/vptree-full.bin"
	if _, err := os.Stat(nullPath); err != nil {
		t.Skip("vptree bins not found")
	}
	forest, err := LoadForest(nullPath, fullPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("null-tree: nodes=%d leaf_vecs=%d", len(forest.nullTree.nodes), len(forest.nullTree.leafLabels))
	t.Logf("full-tree: nodes=%d leaf_vecs=%d", len(forest.fullTree.nodes), len(forest.fullTree.leafLabels))
}
