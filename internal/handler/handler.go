package handler

import (
	"encoding/binary"
	"math"
	"sync"
	"time"

	"encoding/json"

	"github.com/NRLacerda/rinha-de-backend-2026/internal/vectorize"
	"github.com/valyala/fasthttp"
)

// queryTimeout is the hard deadline for each index call.
// Well under k6's 2001ms cutoff — guarantees we always respond, never hang.
const queryTimeout = 200 * time.Millisecond

var reqPool = sync.Pool{New: func() any { return new(vectorize.Request) }}

// Handler calls the index service to get a fraud score.
type Handler struct {
	indexURL string
	client   *fasthttp.Client
}

func New(indexURL string) *Handler {
	return &Handler{
		indexURL: indexURL,
		client:   &fasthttp.Client{MaxConnsPerHost: 64},
	}
}

func (h *Handler) Handle(ctx *fasthttp.RequestCtx) {
	switch string(ctx.Path()) {
	case "/ready":
		// Check that the index service is also ready.
		code, _, err := h.client.Get(nil, "http://index:9000/ready")
		if err != nil || code != fasthttp.StatusOK {
			ctx.SetStatusCode(fasthttp.StatusServiceUnavailable)
			return
		}
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

	if err := json.Unmarshal(ctx.PostBody(), req); err != nil {
		respond(ctx, true, 0.0)
		return
	}

	vec := vectorize.Vectorize(req)

	// Encode vector as 56 bytes (14 × float32 LE).
	var body [56]byte
	for i, v := range vec {
		binary.LittleEndian.PutUint32(body[i*4:], math.Float32bits(v))
	}

	// Query index service.
	fReq := fasthttp.AcquireRequest()
	fResp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(fReq)
	defer fasthttp.ReleaseResponse(fResp)
	fReq.SetRequestURI(h.indexURL)
	fReq.Header.SetMethod("POST")
	fReq.Header.SetContentType("application/octet-stream")
	fReq.SetBody(body[:])
	doErr := h.client.DoTimeout(fReq, fResp, queryTimeout)
	code := fResp.StatusCode()
	respBody := fResp.Body()
	if doErr != nil || code != fasthttp.StatusOK || len(respBody) < 4 {
		// On timeout/error assume fraud: FP (weight 1) < FN (weight 3) < HTTP error (weight 5).
		respond(ctx, false, 1.0)
		return
	}

	bits := binary.LittleEndian.Uint32(respBody[:4])
	score := math.Float32frombits(bits)
	respond(ctx, score < 0.6, score)
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

func respond(ctx *fasthttp.RequestCtx, approved bool, score float32) {
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
