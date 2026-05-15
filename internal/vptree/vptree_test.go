package vptree

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

func randomRecord(rng *rand.Rand) Record {
	var r Record
	for i := range r.Vector {
		r.Vector[i] = rng.Float32()
	}
	if rng.Intn(2) == 0 {
		r.Label = 1
	}
	return r
}

// bruteForceScore is the reference implementation for test comparison.
func bruteForceScore(records []Record, q [dims]float32) float32 {
	type pair struct {
		d float32
		l uint8
	}
	top := make([]pair, 0, k)

	worst := func() float32 {
		if len(top) < k {
			return math.MaxFloat32
		}
		max := top[0].d
		for _, p := range top[1:] {
			if p.d > max {
				max = p.d
			}
		}
		return max
	}

	for _, rec := range records {
		d := dist(q[:], rec.Vector[:])
		w := worst()
		if d < w || len(top) < k {
			if len(top) == k {
				// Replace worst.
				maxIdx := 0
				for i, p := range top {
					if p.d > top[maxIdx].d {
						maxIdx = i
					}
				}
				top[maxIdx] = pair{d, rec.Label}
			} else {
				top = append(top, pair{d, rec.Label})
			}
		}
	}

	var frauds float32
	for _, p := range top {
		frauds += float32(p.l)
	}
	return frauds / float32(k)
}

func TestQueryMatchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	const n = 1000

	records := make([]Record, n)
	for i := range records {
		records[i] = randomRecord(rng)
	}

	tree := Build(records)

	for trial := 0; trial < 50; trial++ {
		var q [dims]float32
		for i := range q {
			q[i] = rng.Float32()
		}

		got := tree.Query(q)
		want := bruteForceScore(records, q)

		if got != want {
			t.Errorf("trial %d: vptree=%.1f brute=%.1f", trial, got, want)
		}
	}
}

func TestSaveLoad(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	const n = 500

	records := make([]Record, n)
	for i := range records {
		records[i] = randomRecord(rng)
	}

	tree := Build(records)

	tmp, err := os.CreateTemp("", "vptree-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	tmp.Close()
	defer os.Remove(tmp.Name())

	if err := tree.Save(tmp.Name()); err != nil {
		t.Fatal("save:", err)
	}

	loaded, err := Load(tmp.Name())
	if err != nil {
		t.Fatal("load:", err)
	}

	rng2 := rand.New(rand.NewSource(42))
	for trial := 0; trial < 20; trial++ {
		var q [dims]float32
		for i := range q {
			q[i] = rng2.Float32()
		}
		orig := tree.Query(q)
		got := loaded.Query(q)
		if orig != got {
			t.Errorf("trial %d: original=%.1f loaded=%.1f", trial, orig, got)
		}
	}
}

func BenchmarkQuery(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const n = 100_000

	records := make([]Record, n)
	for i := range records {
		records[i] = randomRecord(rng)
	}
	tree := Build(records)

	var q [dims]float32
	for i := range q {
		q[i] = rng.Float32()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tree.Query(q)
	}
}
