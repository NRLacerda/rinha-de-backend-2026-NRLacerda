# ---- Build stage ------------------------------------------------------------
FROM golang:1.22-bookworm AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build the index at image-build time: references.json → hnsw.bin
# (references.json must be present in the build context)
RUN go run ./cmd/build-index

# Compile both binaries
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /api       ./cmd/api
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /index-svc ./cmd/index-service

# ---- API image --------------------------------------------------------------
FROM scratch AS api
COPY --from=builder /api /api
EXPOSE 8080
ENTRYPOINT ["/api"]

# ---- Index service image ----------------------------------------------------
FROM debian:bookworm-slim AS index
WORKDIR /app
COPY --from=builder /index-svc        /app/index-svc
COPY --from=builder /app/resources/hnsw.bin /app/resources/hnsw.bin
EXPOSE 9000
ENTRYPOINT ["/app/index-svc"]
