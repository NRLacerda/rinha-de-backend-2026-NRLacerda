package main

import (
	"log"

	"github.com/NRLacerda/rinha-de-backend-2026/internal/vptree"
)

func main() {
	log.Println("loading references.json...")
	records, err := vptree.LoadJSON("resources/references.json")
	if err != nil {
		log.Fatalf("failed to load dataset: %v", err)
	}
	log.Printf("loaded %d records", len(records))

	var nullPool, fullPool []vptree.Record
	for _, r := range records {
		if r.Vector[5] == -1 {
			nullPool = append(nullPool, r)
		} else {
			fullPool = append(fullPool, r)
		}
	}
	log.Printf("null-pool=%d full-pool=%d", len(nullPool), len(fullPool))

	log.Println("building null-pool tree...")
	treeNull := vptree.Build(nullPool)
	if err := treeNull.Save("resources/vptree-null.bin"); err != nil {
		log.Fatalf("save null tree: %v", err)
	}
	log.Println("null tree saved")

	log.Println("building full-pool tree...")
	treeFull := vptree.Build(fullPool)
	if err := treeFull.Save("resources/vptree-full.bin"); err != nil {
		log.Fatalf("save full tree: %v", err)
	}
	log.Println("full tree saved")
}
