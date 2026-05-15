package vptree

import (
	"encoding/json"
	"fmt"
	"os"
)

type jsonRecord struct {
	Vector [dims]float32 `json:"vector"`
	Label  string        `json:"label"`
}

// LoadJSON streams references.json into a []Record without loading the full
// JSON into memory at once. The file is a single JSON array.
func LoadJSON(path string) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	dec := json.NewDecoder(f)

	// Read opening '['.
	if _, err := dec.Token(); err != nil {
		return nil, fmt.Errorf("read '[': %w", err)
	}

	records := make([]Record, 0, 3_000_000)
	var jr jsonRecord
	for dec.More() {
		if err := dec.Decode(&jr); err != nil {
			return nil, fmt.Errorf("decode record: %w", err)
		}
		r := Record{Vector: jr.Vector}
		if jr.Label == "fraud" {
			r.Label = 1
		}
		records = append(records, r)
	}

	return records, nil
}
