package handler

import (
	"sync"

	"github.com/NRLacerda/rinha-de-backend-2026/internal/hnsw"
	"github.com/NRLacerda/rinha-de-backend-2026/internal/vectorize"
	"github.com/bytedance/sonic"
	"github.com/valyala/fasthttp"
)

var reqPool = sync.Pool{New: func() any { return new(vectorize.Request) }}

// Handler serves fraud-score requests using an in-process HNSW index.
type Handler struct {
	index  *hnsw.HNSW
	labels []uint8
}

func New(index *hnsw.HNSW, labels []uint8) *Handler {
	return &Handler{index: index, labels: labels}
}

func (h *Handler) Handle(ctx *fasthttp.RequestCtx) {
	switch string(ctx.Path()) {
	case "/ready":
		ctx.SetStatusCode(fasthttp.StatusOK)
	case "/fraud-score":
		h.fraudScore(ctx)
	default:
		ctx.SetStatusCode(fasthttp.StatusNotFound)
	}
}

func (h *Handler) fraudScore(ctx *fasthttp.RequestCtx) {
	if !ctx.IsPost() {
		ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		return
	}

	req := reqPool.Get().(*vectorize.Request)
	defer func() {
		*req = vectorize.Request{}
		reqPool.Put(req)
	}()

	if err := sonic.Unmarshal(ctx.PostBody(), req); err != nil {
		respond(ctx, 0.0)
		return
	}

	vec := vectorize.Vectorize(req)
	score := h.index.QueryFast5(vec, h.labels)
	respond(ctx, score)
}

// Pre-built response bodies for all 6 possible fraud scores.
var responses = func() [6][]byte {
	scores := [6]string{"0.0", "0.2", "0.4", "0.6", "0.8", "1.0"}
	approved := [6]string{"true", "true", "true", "false", "false", "false"}
	var r [6][]byte
	for i := range r {
		r[i] = []byte(`{"approved":` + approved[i] + `,"fraud_score":` + scores[i] + `}`)
	}
	return r
}()

func respond(ctx *fasthttp.RequestCtx, score float32) {
	idx := int(score*5 + 0.5)
	if idx < 0 {
		idx = 0
	} else if idx > 5 {
		idx = 5
	}
	ctx.SetContentType("application/json")
	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetBody(responses[idx])
}
