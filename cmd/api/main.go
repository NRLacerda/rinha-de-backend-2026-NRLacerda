package main

import (
	"log"
	"runtime"

	"github.com/NRLacerda/rinha-de-backend-2026/internal/handler"
	"github.com/NRLacerda/rinha-de-backend-2026/internal/hnsw"
	"github.com/valyala/fasthttp"
)

func main() {
	runtime.GOMAXPROCS(2)

	index, labels, err := hnsw.Load("resources/hnsw.bin")
	if err != nil {
		log.Fatalf("load: %v", err)
	}

	h := handler.New(index, labels)
	if err := fasthttp.ListenAndServe(":8080", h.Handle); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
