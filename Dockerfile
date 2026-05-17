# ---- Build stage ------------------------------------------------------------
FROM golang:1.22-bookworm AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build the index at image-build time: references.json → hnsw.bin
# (references.json must be present in the build context)
RUN go run ./cmd/build-index

# Compile API binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /api ./cmd/api

# ---- API image --------------------------------------------------------------
FROM debian:bookworm-slim AS api
WORKDIR /app
COPY --from=builder /api                    /app/api
COPY --from=builder /app/resources/hnsw.bin /app/resources/hnsw.bin
EXPOSE 8080
ENTRYPOINT ["/app/api"]
