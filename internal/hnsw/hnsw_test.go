package hnsw

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

func randomVec(rng *rand.Rand) [Dims]float32 {
	var v [Dims]float32
	for i := range v {
		v[i] = rng.Float32()
	}
	return v
}

func bruteForceScore(vecs []float32, labels []uint8, q [Dims]float32) float32 {
	n := len(labels)
	best := [5]struct {
		d float32
		l uint8
	}{{math.MaxFloat32, 0}, {math.MaxFloat32, 0}, {math.MaxFloat32, 0}, {math.MaxFloat32, 0}, {math.MaxFloat32, 0}}
	worstIdx := func() int {
		wi := 0
		for i := 1; i < 5; i++ {
			if best[i].d > best[wi].d {
				wi = i
			}
		}
		return wi
	}
	for i := 0; i < n; i++ {
		v := vecs[i*Dims : i*Dims+Dims]
		d := distVec(q[:], v)
		wi := worstIdx()
		if d < best[wi].d {
			best[wi].d = d
			best[wi].l = labels[i]
		}
	}
	var frauds float32
	for _, b := range best {
		frauds += float32(b.l)
	}
	return frauds / 5
}

func TestHNSWRecall(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	const n = 5000

	vecs := make([]float32, n*Dims)
	labels := make([]uint8, n)
	for i := 0; i < n; i++ {
		v := randomVec(rng)
		copy(vecs[i*Dims:], v[:])
		if rng.Intn(3) == 0 {
			labels[i] = 1
		}
	}

	idx := New(n, 8, 200)
	for i := 0; i < n; i++ {
		idx.AddVector(int32(i), vecs[i*Dims:i*Dims+Dims])
	}
	for i := 0; i < n; i++ {
		idx.Insert(int32(i))
	}

	const trials = 200
	matches := 0
	for trial := 0; trial < trials; trial++ {
		q := randomVec(rng)
		got := idx.Query(q, labels, 50)
		want := bruteForceScore(vecs, labels, q)
		if got == want {
			matches++
		}
	}
	recall := float64(matches) / trials
	t.Logf("exact score match rate: %.1f%% (%d/%d)", recall*100, matches, trials)
	if recall < 0.85 {
		t.Errorf("recall too low: %.1f%% < 85%%", recall*100)
	}
}

func TestSaveLoad(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	const n = 500

	vecs := make([]float32, n*Dims)
	labels := make([]uint8, n)
	for i := 0; i < n; i++ {
		v := randomVec(rng)
		copy(vecs[i*Dims:], v[:])
		labels[i] = uint8(rng.Intn(2))
	}

	idx := New(n, 8, 200)
	for i := 0; i < n; i++ {
		idx.AddVector(int32(i), vecs[i*Dims:i*Dims+Dims])
		idx.Insert(int32(i))
	}

	tmp, err := os.CreateTemp("", "hnsw-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	tmp.Close()
	defer os.Remove(tmp.Name())

	if err := idx.SaveWithLabels(tmp.Name(), labels); err != nil {
		t.Fatal("save:", err)
	}

	loaded, loadedLabels, err := Load(tmp.Name())
	if err != nil {
		t.Fatal("load:", err)
	}

	rng2 := rand.New(rand.NewSource(1))
	for trial := 0; trial < 50; trial++ {
		q := randomVec(rng2)
		orig := idx.Query(q, labels, 50)
		got := loaded.Query(q, loadedLabels, 50)
		if orig != got {
			t.Errorf("trial %d: original=%.1f loaded=%.1f", trial, orig, got)
		}
	}
}

func BenchmarkQuery(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const n = 50_000

	vecs := make([]float32, n*Dims)
	labels := make([]uint8, n)
	for i := 0; i < n; i++ {
		v := randomVec(rng)
		copy(vecs[i*Dims:], v[:])
		labels[i] = uint8(rng.Intn(2))
	}

	idx := New(n, 8, 200)
	for i := 0; i < n; i++ {
		idx.AddVector(int32(i), vecs[i*Dims:i*Dims+Dims])
		idx.Insert(int32(i))
	}

	q := randomVec(rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Query(q, labels, 50)
	}
}
