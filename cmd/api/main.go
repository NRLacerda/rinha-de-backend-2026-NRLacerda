package main

import (
	"log"
	"runtime"

	"github.com/NRLacerda/rinha-de-backend-2026/internal/handler"
	"github.com/NRLacerda/rinha-de-backend-2026/internal/vptree"
	"github.com/valyala/fasthttp"
)

func main() {
	runtime.GOMAXPROCS(1)

	forest, err := vptree.LoadForest("resources/vptree-null.bin", "resources/vptree-full.bin")
	if err != nil {
		log.Fatalf("failed to load vptree forest: %v", err)
	}

	h := handler.New(forest)

	log.Println("listening on :8080")
	if err := fasthttp.ListenAndServe(":8080", h.Handle); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
