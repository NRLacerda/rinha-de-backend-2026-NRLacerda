package main

import (
	"log"
	"runtime"
	"time"

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
	srv := &fasthttp.Server{
		Handler:      h.Handle,
		ReadTimeout:  2 * time.Second,
		WriteTimeout: 2 * time.Second,
	}
	if err := srv.ListenAndServe(":8080"); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
