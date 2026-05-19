# ---- Build stage ------------------------------------------------------------
FROM golang:1.22-bookworm AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Compile API binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /api ./cmd/api

# ---- API image --------------------------------------------------------------
FROM debian:bookworm-slim AS api
WORKDIR /app
COPY --from=builder /api                /app/api
COPY resources/hnsw.bin                 /app/resources/hnsw.bin
EXPOSE 8080
ENTRYPOINT ["/app/api"]
