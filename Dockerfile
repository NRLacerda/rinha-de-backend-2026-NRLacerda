FROM golang:1.22-bookworm AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build the index from references.json → vptree.bin (one-time, baked into image).
RUN go run ./cmd/build-index

# Build the API binary.
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /api ./cmd/api

# ---- Runtime image ----------------------------------------------------------
FROM debian:bookworm-slim

WORKDIR /app

COPY --from=builder /api /app/api
COPY --from=builder /app/resources/vptree.bin /app/resources/vptree.bin

EXPOSE 8080

CMD ["/app/api"]
