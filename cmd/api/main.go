package main

import (
	"log"
	"runtime"

	"github.com/NRLacerda/rinha-de-backend-2026/internal/handler"
	"github.com/valyala/fasthttp"
)

func main() {
	runtime.GOMAXPROCS(1)
	h := handler.New("http://index:9000/query")
	if err := fasthttp.ListenAndServe(":8080", h.Handle); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
