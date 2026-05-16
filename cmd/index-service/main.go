package main

import (
	"encoding/binary"
	"log"
	"math"
	"runtime"
	"sync/atomic"

	"github.com/NRLacerda/rinha-de-backend-2026/internal/hnsw"
	"github.com/valyala/fasthttp"
)

const efSearch = 100

var (
	index  *hnsw.HNSW
	labels []uint8
	ready  atomic.Bool
)

func main() {
	runtime.GOMAXPROCS(1)

	var err error
	index, labels, err = hnsw.Load("resources/hnsw.bin")
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	ready.Store(true)

	if err := fasthttp.ListenAndServe(":9000", handle); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

func handle(ctx *fasthttp.RequestCtx) {
	switch string(ctx.Path()) {
	case "/ready":
		if ready.Load() {
			ctx.SetStatusCode(fasthttp.StatusOK)
		} else {
			ctx.SetStatusCode(fasthttp.StatusServiceUnavailable)
		}

	case "/query":
		if !ctx.IsPost() {
			ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
			return
		}
		body := ctx.PostBody()
		if len(body) != hnsw.Dims*4 {
			ctx.SetStatusCode(fasthttp.StatusBadRequest)
			return
		}

		var q [hnsw.Dims]float32
		for i := range q {
			bits := binary.LittleEndian.Uint32(body[i*4 : i*4+4])
			q[i] = math.Float32frombits(bits)
		}

		score := index.Query(q, labels, efSearch)

		var resp [4]byte
		binary.LittleEndian.PutUint32(resp[:], math.Float32bits(score))
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetBody(resp[:])

	default:
		ctx.SetStatusCode(fasthttp.StatusNotFound)
	}
}
