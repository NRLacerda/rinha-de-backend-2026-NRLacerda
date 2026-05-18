package hnsw

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"runtime"
)

// SaveWithLabels serializes the HNSW index + labels to a binary file.
//
// Format (little-endian):
//   int32    N, M, M0, EfConstruction, ep, epLevel
//   float32  qScale[Dims]          (14 × 4B = 56B)
//   float32  qZero[Dims]           (14 × 4B = 56B)
//   uint8    vectors[N*Dims]       (42 MB at N=3M — was 168 MB as float32)
//   uint8    nodeLevel[N]
//   int32    conn0[N*M0]
//   uint8    conn0cnt[N]
//   upper entries: {int32 nodeID, uint8 numLevels, per-level: {uint8 cnt, cnt×int32 nbs}}
//   int32    -1 sentinel
//   uint8    labels[N]
func (h *HNSW) SaveWithLabels(path string, labels []uint8) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	bw := bufio.NewWriterSize(f, 1<<20)

	wi32 := func(v int32) error {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], uint32(v))
		_, err := bw.Write(b[:])
		return err
	}
	wu8 := func(v uint8) error { return bw.WriteByte(v) }

	for _, v := range []int32{int32(h.N), int32(h.M), int32(h.M0), int32(h.EfConstruction), h.ep, int32(h.epLevel)} {
		if err := wi32(v); err != nil {
			return err
		}
	}

	if err := binary.Write(bw, binary.LittleEndian, h.qScale); err != nil {
		return err
	}
	if err := binary.Write(bw, binary.LittleEndian, h.qZero); err != nil {
		return err
	}
	// Write only the actual Dims bytes per node (not the stride padding).
	for i := 0; i < h.N; i++ {
		if _, err := bw.Write(h.vectors[i*stride : i*stride+Dims]); err != nil {
			return err
		}
	}
	if _, err := bw.Write(h.nodeLevel[:h.N]); err != nil {
		return err
	}
	if err := binary.Write(bw, binary.LittleEndian, h.conn0[:h.N*h.M0]); err != nil {
		return err
	}
	if _, err := bw.Write(h.conn0cnt[:h.N]); err != nil {
		return err
	}

	for id, uc := range h.upperConns {
		if len(uc) == 0 {
			continue
		}
		if err := wi32(id); err != nil {
			return err
		}
		if err := wu8(uint8(len(uc))); err != nil {
			return err
		}
		for _, lvlNbs := range uc {
			if err := wu8(uint8(len(lvlNbs))); err != nil {
				return err
			}
			if err := binary.Write(bw, binary.LittleEndian, lvlNbs); err != nil {
				return err
			}
		}
	}
	if err := wi32(-1); err != nil {
		return err
	}

	if _, err := bw.Write(labels[:h.N]); err != nil {
		return err
	}
	return bw.Flush()
}

// Load reads a previously saved HNSW index and its labels.
// nodeLevel is freed immediately — not needed for Query.
func Load(path string) (*HNSW, []uint8, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	br := bufio.NewReaderSize(f, 1<<20)

	ri32 := func() (int32, error) {
		var b [4]byte
		if _, err := io.ReadFull(br, b[:]); err != nil {
			return 0, err
		}
		return int32(binary.LittleEndian.Uint32(b[:])), nil
	}
	ru8 := func() (uint8, error) { return br.ReadByte() }

	n32, err := ri32()
	if err != nil {
		return nil, nil, err
	}
	n := int(n32)
	m32, _ := ri32()
	m0_32, _ := ri32()
	efC32, _ := ri32()
	ep, _ := ri32()
	epLevel32, err := ri32()
	if err != nil {
		return nil, nil, err
	}

	m, m0 := int(m32), int(m0_32)
	h := &HNSW{
		M:              m,
		M0:             m0,
		EfConstruction: int(efC32),
		N:              n,
		ep:             ep,
		epLevel:        int(epLevel32),
		vectors:        make([]uint8, n*stride),
		nodeLevel:      make([]uint8, n),
		conn0:          make([]int32, n*m0),
		conn0cnt:       make([]uint8, n),
		upperConns:     make(map[int32][][]int32),
		visitMark:      make([]uint32, n),
	}

	if err := binary.Read(br, binary.LittleEndian, &h.qScale); err != nil {
		return nil, nil, fmt.Errorf("qScale: %w", err)
	}
	if err := binary.Read(br, binary.LittleEndian, &h.qZero); err != nil {
		return nil, nil, fmt.Errorf("qZero: %w", err)
	}
	// On-disk format stores Dims bytes per node; zero-pad to stride in memory.
	tmp := make([]uint8, n*Dims)
	if _, err := io.ReadFull(br, tmp); err != nil {
		return nil, nil, fmt.Errorf("vectors: %w", err)
	}
	for i := 0; i < n; i++ {
		copy(h.vectors[i*stride:], tmp[i*Dims:(i+1)*Dims])
	}
	tmp = nil
	if _, err := io.ReadFull(br, h.nodeLevel); err != nil {
		return nil, nil, fmt.Errorf("nodeLevel: %w", err)
	}
	h.nodeLevel = nil  // not needed for Query
	h.visitMark = nil  // replaced by visitPool; free 12 MB before allocating pool slots
	runtime.GC()
	h.initVisitPool(2, h.N) // 2 concurrent queries × 12 MB = 24 MB, total ~306 MB

	if err := binary.Read(br, binary.LittleEndian, h.conn0); err != nil {
		return nil, nil, fmt.Errorf("conn0: %w", err)
	}
	if _, err := io.ReadFull(br, h.conn0cnt); err != nil {
		return nil, nil, fmt.Errorf("conn0cnt: %w", err)
	}

	for {
		id, err := ri32()
		if err != nil {
			return nil, nil, fmt.Errorf("upper id: %w", err)
		}
		if id == -1 {
			break
		}
		numLevels, _ := ru8()
		uc := make([][]int32, numLevels)
		for l := 0; l < int(numLevels); l++ {
			cnt, _ := ru8()
			nbs := make([]int32, cnt)
			if err := binary.Read(br, binary.LittleEndian, nbs); err != nil {
				return nil, nil, fmt.Errorf("upper nbs: %w", err)
			}
			uc[l] = nbs
		}
		h.upperConns[id] = uc
	}

	labels := make([]uint8, n)
	if _, err := io.ReadFull(br, labels); err != nil {
		return nil, nil, fmt.Errorf("labels: %w", err)
	}

	return h, labels, nil
}
