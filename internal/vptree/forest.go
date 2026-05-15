package vptree

import "fmt"

// Forest holds two VP-Trees partitioned by the null sentinel at dims 5 & 6.
// Queries route to the matching tree first, then optionally cross to the other.
type Forest struct {
	nullTree *Tree // records where vector[5] == -1 (last_transaction: null)
	fullTree *Tree // records where vector[5] != -1
}

func LoadForest(nullPath, fullPath string) (*Forest, error) {
	nt, err := Load(nullPath)
	if err != nil {
		return nil, fmt.Errorf("null tree: %w", err)
	}
	ft, err := Load(fullPath)
	if err != nil {
		return nil, fmt.Errorf("full tree: %w", err)
	}
	return &Forest{nullTree: nt, fullTree: ft}, nil
}

// Query returns fraud_score for the 14-dim query vector.
// Routes to null-tree or full-tree based on dim 5 sentinel.
func (f *Forest) Query(q [dims]float32) float32 {
	var h heap5

	if q[5] == -1 {
		// Query has no last_tx — search null-tree first (homogeneous space).
		f.nullTree.search(q[:], f.nullTree.root, &h)
		// Cross into full-tree only if heap isn't saturated with close matches.
		if h.size < k {
			f.fullTree.search(q[:], f.fullTree.root, &h)
		}
	} else {
		// Query has last_tx — search full-tree first.
		f.fullTree.search(q[:], f.fullTree.root, &h)
		if h.size < k {
			f.nullTree.search(q[:], f.nullTree.root, &h)
		}
	}

	var frauds float32
	for i := 0; i < h.size; i++ {
		frauds += float32(h.labels[i])
	}
	return frauds / float32(k)
}
