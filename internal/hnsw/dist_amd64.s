#include "textflag.h"

// func distSSE2(q *int16, v *uint8) uint32
//
// Computes squared L2 distance between q[0..15] (int16) and v[0..15] (uint8).
// Dimensions 14 and 15 are zero-padded, so they contribute 0 to the sum.
//
// Strategy (SSE2):
//   1. Load v[0..7] and v[8..15] as uint8, zero-extend each to 8×int16.
//   2. Load q[0..7] and q[8..15] as int16.
//   3. Subtract: diff = q - v  (fits in int16: both in [0,255]).
//   4. PMADDWL diff, diff  ->  4xint32 of paired squared-sum.
//   5. PADDL the two halves, then horizontal sum -> scalar uint32.
TEXT ·distSSE2(SB), NOSPLIT, $0-20
    MOVQ q+0(FP), DI        // DI = q (*int16)
    MOVQ v+8(FP), SI        // SI = v (*uint8)

    // Zero register used for byte→int16 zero-extension.
    PXOR    X2, X2

    // Load v[0..7] into the low qword of X0 (high qword auto-zeroed by MOVQ).
    MOVQ    0(SI), X0
    // Load v[8..15] into the low qword of X1.
    MOVQ    8(SI), X1

    // Zero-extend: interleave each byte with a zero byte → 8×uint16 per register.
    PUNPCKLBW X2, X0        // X0 = [v0,v1,...,v7]  as uint16×8
    PUNPCKLBW X2, X1        // X1 = [v8,v9,...,v15] as uint16×8

    // Load q[0..7] and q[8..15] as int16 (16 bytes each).
    MOVOU   0(DI), X3       // X3 = q[0..7]
    MOVOU   16(DI), X4      // X4 = q[8..15]

    // diff = q - v  (signed int16 subtraction; result fits in int16)
    PSUBW   X0, X3          // X3 = q[0..7]  - v[0..7]
    PSUBW   X1, X4          // X4 = q[8..15] - v[8..15]

    // PMADDWL with itself: result[i] = diff[2i]^2 + diff[2i+1]^2  (int16 -> int32)
    PMADDWL X3, X3          // X3 = 4xint32 partial sums [dims 0-7]
    PMADDWL X4, X4          // X4 = 4xint32 partial sums [dims 8-15]

    PADDL   X4, X3          // X3 = combined 4xint32

    // Horizontal sum of 4xint32 in X3 -> scalar.
    MOVO    X3, X5
    PSRLDQ  $8, X5          // X5 = [X3[2], X3[3], 0, 0]
    PADDL   X5, X3          // X3[0] = A+C, X3[1] = B+D
    MOVO    X3, X5
    PSRLDQ  $4, X5          // X5 = [X3[1], ...]
    PADDL   X5, X3          // X3[0] = A+B+C+D

    MOVL    X3, ret+16(FP)
    RET
