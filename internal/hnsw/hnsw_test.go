package hnsw

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

// uniformQuant computes per-dimension min/max quantization params from vecs.
func uniformQuant(vecs []float32, n int) (scale, zero [Dims]float32) {
	var mn, mx [Dims]float32
	for d := range mn {
		mn[d] = vecs[d]
		mx[d] = vecs[d]
	}
	for i := 1; i < n; i++ {
		for d := 0; d < Dims; d++ {
			v := vecs[i*Dims+d]
			if v < mn[d] {
				mn[d] = v
			}
			if v > mx[d] {
				mx[d] = v
			}
		}
	}
	for d := 0; d < Dims; d++ {
		r := mx[d] - mn[d]
		if r == 0 {
			r = 1
		}
		scale[d] = r / 255.0
		zero[d] = mn[d]
	}
	return
}

func euclidean(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return float32(math.Sqrt(float64(s)))
}

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
		d := euclidean(q[:], v)
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
	idx.SetQuantization(uniformQuant(vecs, n))
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
		want := bruteForceScore(idx.Vectors(), labels, q)
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
	idx.SetQuantization(uniformQuant(vecs, n))
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
	idx.SetQuantization(uniformQuant(vecs, n))
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

func BenchmarkQueryFast5(b *testing.B) {
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
	idx.SetQuantization(uniformQuant(vecs, n))
	for i := 0; i < n; i++ {
		idx.AddVector(int32(i), vecs[i*Dims:i*Dims+Dims])
		idx.Insert(int32(i))
	}

	q := randomVec(rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.QueryFast5(q, labels)
	}
}

// TestRecallFull measures exact-score match rate against brute force on the 3M index.
// Uses vectors sampled from the reference set itself as queries — same distribution
// as real contest requests, unlike synthetic random vectors.
func TestRecallFull(t *testing.T) {
	if os.Getenv("HNSW_FULL_RECALL") != "1" {
		t.Skip("set HNSW_FULL_RECALL=1 to run full-index recall test")
	}

	const binPath = "../../resources/hnsw.bin"
	if _, err := os.Stat(binPath); err != nil {
		t.Skip("hnsw.bin not found — run cmd/build-index first")
	}

	t.Log("loading hnsw.bin...")
	idx, labels, err := Load(binPath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Logf("loaded %d nodes", idx.N)

	// Dequantize all vectors once for brute force comparisons.
	vecs := idx.Vectors()

	// Sample query vectors from the reference set — same distribution as contest.
	rng := rand.New(rand.NewSource(42))
	const trials = 200
	queryIDs := make([]int, trials)
	for i := range queryIDs {
		queryIDs[i] = rng.Intn(idx.N)
	}

	for _, ef := range []int{10, 20, 50, 100, 200} {
		matches := 0
		for _, qid := range queryIDs {
			// Build float32 query from the sampled reference vector.
			var q [Dims]float32
			copy(q[:], vecs[qid*Dims:qid*Dims+Dims])

			got := idx.Query(q, labels, ef)
			want := bruteForceScore(vecs, labels, q)
			if got == want {
				matches++
			}
		}
		recall := float64(matches) / trials * 100
		t.Logf("ef=%d  recall=%.1f%%  (%d/%d)", ef, recall, matches, trials)
		if ef == 100 && recall < 80 {
			t.Errorf("ef=100 recall too low: %.1f%% < 80%%", recall)
		}
	}
}

// BenchmarkQueryFull loads the real 3M hnsw.bin (if present) and measures query latency.
// Run from the repo root: go test ./internal/hnsw/ -bench=BenchmarkQueryFull -benchmem -benchtime=10s
func BenchmarkQueryFull(b *testing.B) {
	const binPath = "../../resources/hnsw.bin"
	if _, err := os.Stat(binPath); err != nil {
		b.Skip("hnsw.bin not found — run cmd/build-index first")
	}

	b.Log("loading hnsw.bin...")
	idx, labels, err := Load(binPath)
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	b.Logf("loaded %d nodes", idx.N)

	rng := rand.New(rand.NewSource(7))
	q := randomVec(rng)

	for _, ef := range []int{10, 20, 50, 100, 200} {
		ef := ef
		b.Run(fmt.Sprintf("ef=%d", ef), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx.Query(q, labels, ef)
			}
		})
	}

	b.Run("fast5", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			idx.QueryFast5(q, labels)
		}
	})
}
