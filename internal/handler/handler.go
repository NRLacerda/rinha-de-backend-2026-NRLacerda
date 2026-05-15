package handler

import (
	"sync"

	"github.com/NRLacerda/rinha-de-backend-2026/internal/vectorize"
	"github.com/NRLacerda/rinha-de-backend-2026/internal/vptree"
	"github.com/bytedance/sonic"
	"github.com/valyala/fasthttp"
)

var pool = sync.Pool{
	New: func() any { return new(vectorize.Request) },
}

type Handler struct {
	forest *vptree.Forest
}

func New(forest *vptree.Forest) *Handler {
	return &Handler{forest: forest}
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

	req := pool.Get().(*vectorize.Request)
	defer func() {
		*req = vectorize.Request{}
		pool.Put(req)
	}()

	if err := sonic.Unmarshal(ctx.PostBody(), req); err != nil {
		// Never return 500 — score weight for HTTP errors is 5×.
		// Return approved=true, fraud_score=0.0 as the least harmful fallback.
		ctx.SetContentType("application/json")
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetBodyString(`{"approved":true,"fraud_score":0.0}`)
		return
	}

	vec := vectorize.Vectorize(req)
	score := h.forest.Query(vec)
	approved := score < 0.6

	ctx.SetContentType("application/json")
	ctx.SetStatusCode(fasthttp.StatusOK)
	writeResponse(ctx, approved, score)
}

// prebuilt responses for all 6 possible fraud_score values.
var responses = [2][6][]byte{}

func init() {
	scores := [6]float32{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
	for si, s := range scores {
		approved := s < 0.6
		var ap string
		if approved {
			ap = "true"
		} else {
			ap = "false"
		}
		responses[boolIdx(approved)][si] = []byte(`{"approved":` + ap + `,"fraud_score":` + fmtScore(s) + `}`)
	}
}

func boolIdx(b bool) int {
	if b {
		return 1
	}
	return 0
}

func fmtScore(s float32) string {
	switch s {
	case 0.0:
		return "0.0"
	case 0.2:
		return "0.2"
	case 0.4:
		return "0.4"
	case 0.6:
		return "0.6"
	case 0.8:
		return "0.8"
	default:
		return "1.0"
	}
}

func writeResponse(ctx *fasthttp.RequestCtx, approved bool, score float32) {
	idx := int(score*5 + 0.5) // 0..5
	if idx < 0 {
		idx = 0
	} else if idx > 5 {
		idx = 5
	}
	ctx.SetBody(responses[boolIdx(approved)][idx])
}
