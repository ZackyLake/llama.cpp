#if !defined(DATA_A_F32) && !defined(DATA_A_F16)
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#endif

#include "types.glsl"

#if defined(DATA_A_F32)
FLOAT_TYPE dequantize1(uint ib, uint iqs, uint a_offset) {
    return data_a[a_offset + ib];
}
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(data_a[a_offset + ib], data_a[a_offset + ib + 1]);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    return vec4(data_a[a_offset + ib    ], data_a[a_offset + ib + 1],
                data_a[a_offset + ib + 2], data_a[a_offset + ib + 3]);
}
vec4 dequantize4_2aligned(uint ib, uint iqs, uint a_offset) {
    return vec4(data_a[a_offset + ib    ], data_a[a_offset + ib + 1],
                data_a[a_offset + ib + 2], data_a[a_offset + ib + 3]);
}

#endif

#if defined(DATA_A_F16)
FLOAT_TYPE dequantize1(uint ib, uint iqs, uint a_offset) {
    return data_a[a_offset + ib];
}
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(data_a[a_offset + ib], data_a[a_offset + ib + 1]);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    return vec4(data_a[a_offset + ib    ], data_a[a_offset + ib + 1],
                data_a[a_offset + ib + 2], data_a[a_offset + ib + 3]);
}
vec4 dequantize4_2aligned(uint ib, uint iqs, uint a_offset) {
    const vec2 a = data_a_packed32[(a_offset + ib)/2];
    const vec2 b = data_a_packed32[(a_offset + ib)/2 + 1];
    return vec4(a, b);
}
#endif

#if defined(DATA_A_BF16)
FLOAT_TYPE dequantize1(uint ib, uint iqs, uint a_offset) {
    return bf16_to_fp32(data_a[a_offset + ib]);
}
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(bf16_to_fp32(data_a[a_offset + ib]), bf16_to_fp32(data_a[a_offset + ib + 1]));
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    return vec4(bf16_to_fp32(data_a[a_offset + ib    ]), bf16_to_fp32(data_a[a_offset + ib + 1]),
                bf16_to_fp32(data_a[a_offset + ib + 2]), bf16_to_fp32(data_a[a_offset + ib + 3]));
}
vec4 dequantize4_2aligned(uint ib, uint iqs, uint a_offset) {
    const uint a = data_a_packed32[(a_offset + ib)/2];
    const uint b = data_a_packed32[(a_offset + ib)/2 + 1];
    return vec4(uintBitsToFloat((a & 0x0000ffff) << 16),
                uintBitsToFloat( a & 0xffff0000),
                uintBitsToFloat((b & 0x0000ffff) << 16),
                uintBitsToFloat( b & 0xffff0000));
}
#endif

#if defined(DATA_A_Q4_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return (vec2(vui & 0xF, vui >> 4) - 8.0f);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return (vec4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12) - 8.0f);
}
#endif

#if defined(DATA_A_Q4_1)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2(vui & 0xF, vui >> 4);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return vec4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12);
}
#endif

#if defined(DATA_A_Q5_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = uint(data_a[a_offset + ib].qh[1]) << 16 | data_a[a_offset + ib].qh[0];
    const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return (vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y) - 16.0f);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = uint(data_a_packed16[a_offset + ib].qh[1]) << 16 | data_a_packed16[a_offset + ib].qh[0];
    const ivec2 qh0 = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const ivec2 qh1 = ivec2(((uint_qh >> (iqs + 1)) << 4) & 0x10, (uint_qh >> (iqs + 13)) & 0x10);
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return (vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y) - 16.0f);
}
#endif

#if defined(DATA_A_Q5_1)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = data_a[a_offset + ib].qh;
    const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = data_a_packed16[a_offset + ib].qh;
    const ivec2 qh0 = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const ivec2 qh1 = ivec2(((uint_qh >> (iqs + 1)) << 4) & 0x10, (uint_qh >> (iqs + 13)) & 0x10);
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y);
}
#endif

#if defined(DATA_A_Q8_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(int(data_a[a_offset + ib].qs[iqs]), int(data_a[a_offset + ib].qs[iqs + 1]));
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const i8vec2 v0 = unpack8(int32_t(data_a_packed16[a_offset + ib].qs[iqs/2])).xy; // vec4 used due to #12147
    const i8vec2 v1 = unpack8(int32_t(data_a_packed16[a_offset + ib].qs[iqs/2 + 1])).xy;
    return vec4(v0.x, v0.y, v1.x, v1.y);
}
#endif

#if defined(DATA_A_Q1_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint bits = uint(data_a[a_offset + ib].qs[iqs / 8u]) >> (iqs % 8u);
    return vec2(
        (bits & 1u) != 0u ? 1.0f : -1.0f,
        (bits & 2u) != 0u ? 1.0f : -1.0f);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint bits = uint(data_a[a_offset + ib].qs[iqs / 8u]) >> (iqs % 8u);
    return vec4(
        (bits & 1u) != 0u ? 1.0f : -1.0f,
        (bits & 2u) != 0u ? 1.0f : -1.0f,
        (bits & 4u) != 0u ? 1.0f : -1.0f,
        (bits & 8u) != 0u ? 1.0f : -1.0f);
}
#endif

#if defined(DATA_A_Q2_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint bits = uint(data_a[a_offset + ib].qs[iqs / 4u]) >> (2u * (iqs % 4u));
    return vec2(bits & 3u, (bits >> 2u) & 3u) - 1.0f;
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint bits = uint(data_a[a_offset + ib].qs[iqs / 4u]);
    return vec4(bits & 3u, (bits >> 2u) & 3u, (bits >> 4u) & 3u, bits >> 6u) - 1.0f;
}
#endif

#if defined(DATA_A_IQ1_S)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;
    const int i8 = int(iqs % 8);
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const float dl = float(2 * bitfieldExtract(qh, 12, 3) + 1);
    const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
    const uint idxhi = bitfieldExtract(qh, 3 * int(ib8 & 3), 3);
    const int16_t grid = int16_t(iq1s_grid[qs | (idxhi << 8)]);
    // Signed bitfield extract.
    const ivec2 gvec = ivec2(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2)
    );
    return dl * (vec2(gvec) + delta);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;
    const int i8 = int(iqs % 8);
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const float dl = 2 * bitfieldExtract(qh, 12, 3) + 1;
    const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
    const int16_t grid = int16_t(iq1s_grid[qs | (bitfieldExtract(qh, 3 * int(ib8 & 3), 3) << 8)]);
    // Signed bitfield extract.
    const ivec4 gvec = ivec4(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2),
      bitfieldExtract(grid, 2 * (i8 + 2), 2),
      bitfieldExtract(grid, 2 * (i8 + 3), 2)
    );
    return dl * (vec4(gvec) + delta);
}
#endif

#if defined(DATA_A_IQ1_M)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib8 = iqs / 8;
    const uint ib16 = iqs / 16;
    const int i8 = int(iqs % 8);
    const uint sc = data_a[a_offset + ib].scales[iqs / 64];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib16] >> (4 * (ib8 & 1));
    const float dl = 2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1;
    const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;
    const int16_t grid = int16_t(iq1s_grid[qs | ((qh & 7) << 8)]);
    // Signed bitfield extract.
    const ivec2 gvec = ivec2(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2)
    );
    return dl * (vec2(gvec) + delta);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib8 = iqs / 8;
    const uint ib16 = iqs / 16;
    const int i8 = int(iqs % 8);
    const uint sc = data_a[a_offset + ib].scales[iqs / 64];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib16] >> (4 * (ib8 & 1));
    const float dl = 2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1;
    const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;
    const int16_t grid = int16_t(iq1s_grid[qs | ((qh & 7) << 8)]);
    // Signed bitfield extract.
    const ivec4 gvec = ivec4(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2),
      bitfieldExtract(grid, 2 * (i8 + 2), 2),
      bitfieldExtract(grid, 2 * (i8 + 3), 2)
    );
    return dl * (vec4(gvec) + delta);
}
#endif

#if defined(DATA_A_IQ2_XXS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = (iqs / 8) % 4;
    const uint qs = data_a[a_offset + ib].qs[8 * ib32 + ib8];
    // Scales are stored as packed 7+7+7+7+4 bits (4 sign tuples and 1 int4 scale)
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[4 * ib32 + 2],
        data_a_packed16[a_offset + ib].qs[4 * ib32 + 3]));
    const float db = 0.25 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * int(ib8), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xxs_grid[qs][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = (iqs / 8) % 4;
    const uint qs = data_a[a_offset + ib].qs[8 * ib32 + ib8];
    // Scales are stored as packed 7+7+7+7+4 bits (4 sign tuples and 1 int4 scale)
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[4 * ib32 + 2],
        data_a_packed16[a_offset + ib].qs[4 * ib32 + 3]));
    const float db = 0.25 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * int(ib8), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xxs_grid[qs][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ2_XS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint scale = (data_a[a_offset + ib].scales[iqs / 32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[iqs / 8];
    const float db = 0.25 * (0.5 + scale);
    const uint sign7 = qs >> 9;
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xs_grid[qs & 511][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint scale = (data_a[a_offset + ib].scales[iqs / 32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[iqs / 8];
    const float db = 0.25 * (0.5 + scale);
    const uint sign7 = qs >> 9;
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xs_grid[qs & 511][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ2_S)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;

    const uint scale = (data_a[a_offset + ib].scales[ib32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qhshift = 2 * (ib8 % 4);
    const uint sign = data_a[a_offset + ib].qs[QUANT_K / 8 + ib8] >> (iqs % 8);

    const float db = 0.25 * (0.5 + scale);
    const u8vec4 grid = unpack8(iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)][(iqs % 8) / 4]);
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid[iqs % 4] * (sign0 ? -1.0 : 1.0),
        grid[(iqs % 4) + 1] * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;

    const uint scale = (data_a[a_offset + ib].scales[ib32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qhshift = 2 * (ib8 % 4);
    const uint sign = data_a[a_offset + ib].qs[QUANT_K / 8 + ib8] >> (iqs % 8);

    const float db = 0.25 * (0.5 + scale);
    const u8vec4 grid = unpack8(iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)][(iqs % 8) / 4]);
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ3_XXS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib4 = iqs / 4;
    const uint ib32 = iqs / 32;
    const uint is = QUANT_K / 4 + 4 * ib32;
    const uint qs = data_a[a_offset + ib].qs[ib4];
    // Scales are stored as packed 7+7+7+7+4 bits (4 sign tuples and 1 int4 scale)
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[is / 2],
        data_a_packed16[a_offset + ib].qs[is / 2 + 1]));
    const float db = 0.5 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * (int(ib4 / 2) % 4), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq3xxs_grid[qs] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib4 = iqs / 4;
    const uint ib32 = iqs / 32;
    const uint is = QUANT_K / 4 + 4 * ib32;
    const uint qs = data_a[a_offset + ib].qs[ib4];
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[is / 2],
        data_a_packed16[a_offset + ib].qs[is / 2 + 1]));
    const float db = 0.5 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * (int(ib4 / 2) % 4), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq3xxs_grid[qs]);
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ3_S)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint qs = data_a[a_offset + ib].qs[iqs / 4];
    const uint qh = data_a[a_offset + ib].qh[iqs / 32];
    const uint sign = data_a[a_offset + ib].signs[iqs / 8] >> (iqs % 8);
    const uint scale = data_a[a_offset + ib].scales[iqs / 64];
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    const float db = 1 + 2 * ((scale >> (4 * ((iqs / 32) & 1))) & 0xf);
    const uint32_t grid = iq3s_grid[qs | ((qh << (8 - ((iqs / 4) % 8))) & 256)] >> (8 * (iqs % 4));
    return db * vec2(
        int(grid & 0xFF) * (sign0 ? -1.0 : 1.0),
        int((grid >> 8) & 0xFF) * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib4 = iqs / 4;
    const uint ib32 = iqs / 32;
    const uint qs = data_a[a_offset + ib].qs[ib4];
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint sign = data_a[a_offset + ib].signs[iqs / 8] >> (iqs % 8);
    const uint scale = data_a[a_offset + ib].scales[ib32 / 2];
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    const float db = 1 + 2 * ((scale >> (4 * (ib32 & 1))) & 0xf);
    const uint32_t grid = iq3s_grid[qs | ((qh << (8 - ib4 % 8)) & 256)] >> (8 * (iqs % 4));
    return db * vec4(
        int(grid & 0xFF) * (sign0 ? -1.0 : 1.0),
        int((grid >> 8) & 0xFF) * (sign1 ? -1.0 : 1.0),
        int((grid >> 16) & 0xFF) * (sign2 ? -1.0 : 1.0),
        int((grid >> 24) & 0xFF) * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ4_XS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint iq = 16 * ib32 + (iqs % 16);

    const uint sl = (data_a[a_offset + ib].scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
    const uint sh = (data_a[a_offset + ib].scales_h >> (2 * ib32)) & 3;
    const uint qshift = (iqs & 16) >> 2;
    u8vec2 qs = u8vec2(data_a[a_offset + ib].qs[iq], data_a[a_offset + ib].qs[iq + 1]);
    qs = (qs >> qshift) & uint8_t(0xF);

    const float dl = float(int(sl | (sh << 4)) - 32);
    return dl * vec2(kvalues_iq4nl[qs.x], kvalues_iq4nl[qs.y]);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint iq = 16 * ib32 + (iqs % 16);

    const uint sl = (data_a[a_offset + ib].scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
    const uint sh = (data_a[a_offset + ib].scales_h >> (2 * ib32)) & 3;
    const uint qshift = (iqs & 16) >> 2;
    const u8vec4 qs = unpack8((data_a_packed32[a_offset + ib].qs[iq/4] >> qshift) & 0x0F0F0F0F);

    const float dl = float(int(sl | (sh << 4)) - 32);
    return dl * vec4(
        kvalues_iq4nl[qs.x], kvalues_iq4nl[qs.y],
        kvalues_iq4nl[qs.z], kvalues_iq4nl[qs.w]);
}
#endif

#if defined(DATA_A_IQ4_NL)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2(kvalues_iq4nl[vui & 0xF], kvalues_iq4nl[vui >> 4]);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return vec4(kvalues_iq4nl[vui & 0xF], kvalues_iq4nl[(vui >> 4) & 0xF], kvalues_iq4nl[(vui >> 8) & 0xF], kvalues_iq4nl[vui >> 12]);
}
#endif

#if defined(DATA_A_MXFP4)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
#ifdef USE_OCP_FP4
    return vec2(unpackFloat2xfe2m1EXT(uint8_t(vui)));
#else
    return vec2(kvalues_mxfp4[vui & 0xF], kvalues_mxfp4[vui >> 4]) * 0.5;
#endif
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
#ifdef USE_OCP_FP4
    const uint16_t vui = uint16_t(uint(data_a[a_offset + ib].qs[iqs]) |
                                  uint(data_a[a_offset + ib].qs[iqs + 1]) << 8);
    return vec4(unpackFloat4xfe2m1EXT(vui));
#else
    vec2 v0 = dequantize(ib, iqs, a_offset);
    vec2 v1 = dequantize(ib, iqs + 1, a_offset);
    return vec4(v0.x, v0.y, v1.x, v1.y);
#endif
}
#endif

#if defined(DATA_A_NVFP4)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint sub = iqs >> 4;
    const float d = ue4m3_to_fp32(data_a[a_offset + ib].d[sub]);
    const uint j = iqs & 7;
    const uint shift = (iqs & 8) >> 1; // 0 or 4
#ifdef USE_OCP_FP4
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[(sub * 8u + j) / 2u]);
    return vec2(bitcastExtractfe2m1EXT(unpack8(vui).xy, shift)) * d;
#else
    const uint vui0 = uint(data_a[a_offset + ib].qs[sub * 8u + j]);
    const uint vui1 = uint(data_a[a_offset + ib].qs[sub * 8u + j + 1]);
    const uint qs0 = (vui0 >> shift) & 0xF;
    const uint qs1 = (vui1 >> shift) & 0xF;
    return vec2(float(kvalues_mxfp4[qs0]), float(kvalues_mxfp4[qs1])) * d * 0.5;
#endif
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
#ifdef USE_OCP_FP4
    const uint sub = iqs >> 4;
    const float d = ue4m3_to_fp32(data_a[a_offset + ib].d[sub]);
    const uint j = iqs & 7;
    const uint shift = (iqs & 8) >> 1; // 0 or 4
    const uint vui = data_a_packed32[a_offset + ib].qs[(sub * 8u + j) / 4u];
    return vec4(bitcastExtractfe2m1EXT(unpack8(vui), shift)) * d;
#else
    const vec2 v0 = dequantize(ib, iqs, a_offset);
    const vec2 v1 = dequantize(ib, iqs + 2u, a_offset);
    return vec4(v0.x, v0.y, v1.x, v1.y);
#endif
}
#endif

#if defined(DATA_A_F32) || defined(DATA_A_F16) || defined(DATA_A_BF16)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(0, 0);
}
#endif

#if defined(DATA_A_IQ1_M)
vec2 get_dm(uint ib, uint a_offset) {
    const uint16_t[4] scales = data_a[a_offset + ib].scales;
    const u16vec4 s = u16vec4(scales[0], scales[1], scales[2], scales[3]) >> 12;
    const float d = float(unpackHalf2x16(s.x | (s.y << 4) | (s.z << 8) | (s.w << 12)).x);
    return vec2(d, 0);
}
#endif

#if defined(DATA_A_Q2_0) || defined(DATA_A_Q4_0) || defined(DATA_A_Q5_0) || defined(DATA_A_Q8_0) || defined(DATA_A_IQ1_S) || defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ2_XS) || defined(DATA_A_IQ2_S) || defined(DATA_A_IQ3_XXS) || defined(DATA_A_IQ3_S) || defined(DATA_A_IQ4_XS) || defined(DATA_A_IQ4_NL)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(float(data_a[a_offset + ib].d), 0);
}
#endif

#if defined(DATA_A_Q1_0)
vec2 get_dm(uint ib, uint a_offset) {
    const float d = float(data_a[a_offset + ib].d);
    return vec2(d, 0);
}
#endif

#if defined(DATA_A_MXFP4)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(e8m0_to_fp32(data_a[a_offset + ib].e), 0);
}
#endif

#if defined(DATA_A_NVFP4)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1.0, 0.0);
}
#endif

#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)
vec2 get_dm(uint ib, uint a_offset) {
    const vec2 dm = vec2(data_a_packed32[a_offset + ib].dm);
    return dm;
}
#endif

#if defined(DATA_A_Q2_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint qsi = (iqs / 64) * 32 + (iqs % 16) * 2; // 0,2,4..30
    const uint scalesi = iqs / 8;                      // 0..15
    const uint qsshift = ((iqs % 64) / 16) * 2;        // 0,2,4,6

    const uvec2 qs = uvec2(data_a[a_offset + ib].qs[qsi], data_a[a_offset + ib].qs[qsi + 1]);
    const uint scales = data_a[a_offset + ib].scales[scalesi];
    const vec2 dm = vec2(data_a[a_offset + ib].dm);

    return dm.x * float(scales & 0xF) * vec2((qs >> qsshift) & 3) - dm.y * float(scales >> 4);
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_IQ2_K) || defined(DATA_A_IQ3_K) || defined(DATA_A_IQ4_K) || defined(DATA_A_IQ5_K) || defined(DATA_A_IQ6_K)
float dequantize_iqk(uint ib, uint element) {
#if defined(DATA_A_IQ2_K)
    const uint ib128 = element / 128;
    const uint pos128 = element % 128;
    const uint group = pos128 / 32;
    const uint pos32 = pos128 % 32;
    const uint il = pos32 / 2;
    const uint j = pos32 % 2;
    const uint q = uint(data_a[ib].qs[32 * ib128 + 2 * il + j]);
    const uint extra = uint(data_a[ib].extra) >> (8 * ib128 + il / 8);
    const uint scale = (uint(data_a[ib].scales[4 * ib128 + group]) >> (4 * (il / 8))) & 0x0F;
    const uint table = (extra >> (2 * group)) & 1;
    return float(data_a[ib].d) * float(int(scale) - 8) * float(kvalues_iq2_k[((q >> (2 * group)) & 3) + 4 * table]);
#elif defined(DATA_A_IQ3_K)
    const uint ib32 = element / 32;
    const uint pos = element % 32;
    const uint subblock = pos / 16;
    const uint j = pos % 16;
    const uint shift_l = 2 * (ib32 % 4);
    const uint shift_h = ib32 % 8;
    const uint q = uint(data_a[ib].qs[32 * (ib32 / 4) + 16 * subblock + j]);
    const uint h = uint(data_a[ib].qh[16 * subblock + j]) >> shift_h;
    const uint scale = (uint(data_a[ib].scales_l[ib32]) >> (4 * subblock)) & 0x0F;
    const uint sign = (uint(data_a[ib].scales_h) >> (2 * ib32 + subblock)) & 1;
    const uint table = (uint(data_a[ib].extra) >> (2 * ib32 + subblock)) & 1;
    const uint index = ((q >> shift_l) & 3) | ((h & 1) << 2) | (table << 3);
    return float(data_a[ib].d) * float(2 * scale + 1) * (sign != 0 ? -1.0 : 1.0) * float(kvalues_iq3_k[index]);
#elif defined(DATA_A_IQ4_K)
    const uint ib32 = element / 32;
    const uint pos32 = element % 32;
    const uint il = (pos32 % 16) / 4;
    const uint j = pos32 % 4;
    const uint subblock = pos32 / 16;
    const uint scale_high = uint(data_a[ib].scales_h[ib32 / 2]) >> (4 * (ib32 % 2));
    const uint scale_low = uint(data_a[ib].scales_l[ib32]);
    const uint scale = subblock == 0 ? ((scale_low & 0x0F) | ((scale_high << 4) & 0x30)) : ((scale_low >> 4) | ((scale_high << 2) & 0x30));
    const uint table = (uint(data_a[ib].extra) >> (2 * ib32 + subblock)) & 1;
    const uint q = uint(data_a[ib].qs[16 * ib32 + 4 * il + j]);
    return float(data_a[ib].d) * float(int(scale) - 32) * float(kvalues_iq4_k[((q >> (4 * subblock)) & 0x0F) + 16 * table]);
#elif defined(DATA_A_IQ5_K)
    const uint ib64 = element / 64;
    const uint pos64 = element % 64;
    const uint group = pos64 / 16;
    const uint j = pos64 % 16;
    const uint q_index = 32 * ib64 + 16 * (group % 2) + j;
    const uint h_index = 16 * (group % 2) + j;
    const uint q = uint(data_a[ib].qs[q_index]);
    const uint h = uint(data_a[ib].qh[h_index]) >> (2 * (ib64 % 4));
    const uint extra = uint(data_a[ib].extra) >> (4 * (ib64 % 4));
    const uint scale_low = uint(data_a[ib].scales_l[2 * ib64 + group / 2]);
    const uint scale_high = uint(data_a[ib].scales_h[ib64]);
    const uint scale = ((scale_low >> (4 * (group % 2))) & 0x0F) | (((scale_high >> (2 * group)) & 3) << 4);
    const uint value = group < 2 ? ((q & 0x0F) | ((h & 1) << 4)) : ((q >> 4) | ((h & 2) << 3));
    const uint index = value | (((extra >> group) & 1) << 5);
    return float(data_a[ib].d) * float(int(scale) - 32) * float(kvalues_iq5_k[index]);
#else
    const uint ib64 = element / 64;
    const uint pos64 = element % 64;
    const uint group = pos64 / 16;
    const uint j = pos64 % 16;
    const uint q_index = 32 * ib64 + 16 * (group % 2) + j;
    const uint h_index = 32 * (ib64 / 2) + 16 * (group % 2) + j;
    const uint q = uint(data_a[ib].qs[q_index]);
    const uint h = uint(data_a[ib].qh[h_index]) >> (4 * (ib64 % 2));
    const uint extra = uint(data_a[ib].extra) >> (4 * (ib64 % 4));
    const uint value = group < 2 ? ((q & 0x0F) | ((h & 3) << 4)) : ((q >> 4) | ((h & 0x0C) << 2));
    const float qvalue = -127.0 + float(value) * (6.2568 + float(value) * (-0.11218 + float(value) * 0.0011972));
    return float(data_a[ib].d) * float(data_a[ib].scales[4 * ib64 + group]) * (qvalue + float((extra >> group) & 1));
#endif
}
#endif

#if defined(DATA_A_IQK_ROW)
uint iqk_row_load_u8(uint offset) {
    return (uint(data_a[offset / 2]) >> (8 * (offset & 1))) & 0xFF;
}

uint iqk_row_load_u16(uint offset) {
    return uint(data_a[offset / 2]);
}

uint iqk_row_load_u32(uint offset) {
    return iqk_row_load_u16(offset) | (iqk_row_load_u16(offset + 2) << 16);
}

float iqk_row_scale(uint row_offset) {
#if defined(DATA_A_IQ2_KS) || defined(DATA_A_IQ2_KL) || defined(DATA_A_IQ3_KS)
    return unpackHalf2x16(iqk_row_load_u16(row_offset)).x;
#else
    return uintBitsToFloat(iqk_row_load_u32(row_offset));
#endif
}

#if defined(DATA_A_IQ1_KT) || defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT) || defined(DATA_A_IQ4_KT)
int iqkt_next(inout uint state) {
    state *= 0xCBAC1FEDu;
    const uint value = state & 0x3F3F3F3Fu;
    return int(value & 0xFF) + int((value >> 8) & 0xFF) + int((value >> 16) & 0xFF) + int(value >> 24) - 126;
}

int iqkt_value(uint block_offset, uint element) {
    const uint group8 = element / 8;
    const uint pos8 = element % 8;
    const uint ib32 = element / 32;
    uint state;

#if defined(DATA_A_IQ1_KT)
    const uint sh = iqk_row_load_u8(block_offset + ib32);
    state = iqk_row_load_u8(block_offset + 8 + group8) |
            ((iqk_row_load_u8(block_offset + 40 + group8 % 16) >> (4 * (group8 / 16))) & 0x0F) << 8 |
            ((sh >> (4 + group8 % 4)) & 1) << 12;
    state += 4096;
#elif defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT)
    state = iqk_row_load_u16(block_offset + 4 + 2 * group8) + 4096;
#else
    const uint header = iqk_row_load_u32(block_offset + 4 * ib32);
    const uint seed_index = 2 * group8 + pos8 / 4;
    state = iqk_row_load_u8(block_offset + 32 + seed_index) |
            ((iqk_row_load_u8(block_offset + 96 + seed_index % 32) >> (4 * (seed_index / 32))) & 0x0F) << 8 |
            ((header >> (8 + 3 * (seed_index % 8))) & 7) << 12 |
            (header & 1) << 15;
    state += 4096;
#endif

    int value = 0;
    const uint count =
#if defined(DATA_A_IQ4_KT)
        pos8 % 4;
#else
        pos8;
#endif
    for (uint j = 0; j <= count; ++j) {
        value = iqkt_next(state);
    }

#if defined(DATA_A_IQ3_KT)
    value = abs(value);
    if (((iqk_row_load_u8(block_offset + 68 + element % 32) >> ib32) & 1) != 0) {
        value = -value;
    }
#endif
    return value;
}

float iqkt_d_scale(uint row_offset, uint block_offset, uint ib32) {
#if defined(DATA_A_IQ1_KT)
    return iqk_row_scale(row_offset) * float(kvalues_iqkt_scale[iqk_row_load_u8(block_offset + ib32) & 0x0F]);
#elif defined(DATA_A_IQ2_KT)
    const uint scale = (iqk_row_load_u8(block_offset + ib32 % 4) >> (4 * (ib32 / 4))) & 0x0F;
    return iqk_row_scale(row_offset) * float(kvalues_iqkt_scale[scale]);
#elif defined(DATA_A_IQ3_KT)
    const uint scale = (iqk_row_load_u8(block_offset + ib32 % 4) >> (4 * (ib32 / 4))) & 0x0F;
    return iqk_row_scale(row_offset) * float(scale);
#else
    return iqk_row_scale(row_offset) * float(int((iqk_row_load_u32(block_offset + 4 * ib32) & 0xFF) >> 1) - 64);
#endif
}
#endif

float dequantize_iqk_row(uint row_offset, uint block_offset, uint element) {
    const uint ib32 = element / 32;
    const uint pos = element % 32;

#if defined(DATA_A_IQ1_KT) || defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT) || defined(DATA_A_IQ4_KT)
    return iqkt_d_scale(row_offset, block_offset, ib32) * float(iqkt_value(block_offset, element));
#elif defined(DATA_A_IQ4_KSS)
    const uint word_offset = block_offset + 4 * (4 * ib32 + (pos % 16) / 4);
    uint values = iqk_row_load_u32(word_offset) & 0xFFFEFFFE;
    values ^= values >> 1;
    const uint index = (values >> (8 * (pos % 4) + 4 * (pos / 16))) & 0x0F;
    uint scale_bits = 0;
    [[unroll]] for (uint j = 0; j < 4; ++j) {
        scale_bits |= (iqk_row_load_u32(block_offset + 4 * (4 * ib32 + j)) & 0x00010001) << (2 * j);
    }
    const uint scale = (scale_bits | (scale_bits >> 15)) & 0xFF;
    return iqk_row_scale(row_offset) * float(int(scale & 254) - 127) * float(kvalues_iq4_kss[index + 16 * (scale & 1)]);
#elif defined(DATA_A_IQ2_KS)
    const uint half_idx = element / 128;
    const uint group = (element % 128) / 32;
    const uint extra = iqk_row_load_u16(block_offset) >> (4 * half_idx);
    const uint packed_scale = iqk_row_load_u8(block_offset + 2 + 2 * half_idx + group / 2);
    const uint scale = ((packed_scale >> (4 * (group % 2))) & 0x0F) | (((extra >> (8 + group)) & 1) << 4);
    const uint value = (iqk_row_load_u8(block_offset + 6 + 32 * half_idx + pos) >> (2 * group)) & 3;
    return iqk_row_scale(row_offset) * float(int(scale) - 16) * float(kvalues_iq2_ks[value + 4 * ((extra >> group) & 1)]);
#elif defined(DATA_A_IQ2_KL)
    const uint ib64 = element / 64;
    const uint pos64 = element % 64;
    const uint group = pos64 / 32;
    const uint pos32 = pos64 % 32;
    const uint pair = pos32 / 2;
    const uint lane = pos32 % 2;
    const uint sh = iqk_row_load_u16(block_offset) >> (4 * ib64);
    const uint scale_index = (2 * ib64 + group) % 4;
    const uint scale_low = (iqk_row_load_u8(block_offset + 2 + scale_index) >> (4 * (ib64 / 2))) & 0x0F;
    const uint scale_high = group == 0 ? ((sh << 4) & 0x30) : ((sh << 2) & 0x30);
    const uint scale = scale_low | scale_high;
    const uint q = iqk_row_load_u8(block_offset + 6 + 16 * ib64 + pair);
    const uint h = iqk_row_load_u8(block_offset + 70 + pair) >> (2 * ib64);
    const uint index = group == 0 ? ((q & 0x0F) | ((h & 1) << 4)) : ((q >> 4) | ((h & 2) << 3));
    return iqk_row_scale(row_offset) * float(int(scale) - 32) * float(kvalues_iq2_kl[2 * index + lane]);
#elif defined(DATA_A_IQ3_KS)
    const uint half_idx = element / 128;
    const uint group = (element % 128) / 32;
    const uint extra = iqk_row_load_u16(block_offset) >> (4 * half_idx);
    const uint scale = ((iqk_row_load_u8(block_offset + 2 + group) >> (4 * half_idx)) & 0x0F) | (((extra >> group) & 1) << 4);
    const uint low = (iqk_row_load_u8(block_offset + 6 + 32 * half_idx + pos) >> (2 * group)) & 3;
    const uint high = (iqk_row_load_u8(block_offset + 70 + pos) >> (4 * half_idx + group)) & 1;
    const uint value = low | (high << 2) | (((extra >> (8 + group)) & 1) << 3);
    return iqk_row_scale(row_offset) * float(int(scale) - 16) * float(kvalues_iq3_ks[value]);
#elif defined(DATA_A_IQ4_KS)
    const uint scale = iqk_row_load_u8(block_offset + ib32);
    const uint value = iqk_row_load_u8(block_offset + 8 + 16 * ib32 + pos % 16);
    const uint index = ((value >> (4 * (pos / 16))) & 0x0F) + 16 * (scale & 1);
    return iqk_row_scale(row_offset) * float(int(scale & 254) - 127) * float(kvalues_iq4_ks[index]);
#else
    const uint ib64 = element / 64;
    const uint pos64 = element % 64;
    const uint scale = iqk_row_load_u8(block_offset + 2 * ib64 + pos64 / 32);
    const uint value = iqk_row_load_u8(block_offset + 8 + 32 * ib64 + pos64 % 32);
    const uint high = (iqk_row_load_u8(block_offset + 136 + pos64 % 32) >> (2 * ib64 + pos64 / 32)) & 1;
    const uint index = ((value >> (4 * (pos64 / 32))) & 0x0F) | (high << 4) | ((scale & 1) << 5);
    return iqk_row_scale(row_offset) * float(int(scale & 254) - 127) * float(kvalues_iq5_ks[index]);
#endif
}

#if defined(DATA_A_IQK_ROW)
vec4 dequantize_iqk_row4(uint row_offset, uint block_offset, uint element) {
#if defined(DATA_A_IQ1_KT) || defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT) || defined(DATA_A_IQ4_KT)
    const uint group8 = element / 8;
    const uint pos8 = element % 8;
    const uint ib32 = element / 32;
    uint state;

#if defined(DATA_A_IQ1_KT)
    const uint sh = iqk_row_load_u8(block_offset + ib32);
    state = iqk_row_load_u8(block_offset + 8 + group8) |
            ((iqk_row_load_u8(block_offset + 40 + group8 % 16) >> (4 * (group8 / 16))) & 0x0F) << 8 |
            ((sh >> (4 + group8 % 4)) & 1) << 12;
    state += 4096;
#elif defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT)
    state = iqk_row_load_u16(block_offset + 4 + 2 * group8) + 4096;
#else
    const uint header = iqk_row_load_u32(block_offset + 4 * ib32);
    const uint seed_index = 2 * group8 + pos8 / 4;
    state = iqk_row_load_u8(block_offset + 32 + seed_index) |
            ((iqk_row_load_u8(block_offset + 96 + seed_index % 32) >> (4 * (seed_index / 32))) & 0x0F) << 8 |
            ((header >> (8 + 3 * (seed_index % 8))) & 7) << 12 |
            (header & 1) << 15;
    state += 4096;
#endif

    const uint count =
#if defined(DATA_A_IQ4_KT)
        pos8 % 4;
#else
        pos8;
#endif
    [[unroll]] for (uint j = 0; j < count; ++j) {
        iqkt_next(state);
    }

    int value0 = iqkt_next(state);
    int value1 = iqkt_next(state);
    int value2 = iqkt_next(state);
    int value3 = iqkt_next(state);

#if defined(DATA_A_IQ3_KT)
    value0 = abs(value0);
    value1 = abs(value1);
    value2 = abs(value2);
    value3 = abs(value3);
    const u8vec4 signs = unpack8(iqk_row_load_u32(block_offset + 68 + element % 32));
    if (((uint(signs.x) >> ib32) & 1) != 0) value0 = -value0;
    if (((uint(signs.y) >> ib32) & 1) != 0) value1 = -value1;
    if (((uint(signs.z) >> ib32) & 1) != 0) value2 = -value2;
    if (((uint(signs.w) >> ib32) & 1) != 0) value3 = -value3;
#endif

    const float d = iqkt_d_scale(row_offset, block_offset, ib32);
    return d * vec4(float(value0), float(value1), float(value2), float(value3));
#elif defined(DATA_A_IQ4_KSS)
    const uint ib32 = element / 32;
    const uint pos = element % 32;
    const uint word_offset = block_offset + 4 * (4 * ib32 + (pos % 16) / 4);
    uint values = iqk_row_load_u32(word_offset) & 0xFFFEFFFE;
    values ^= values >> 1;

    uint scale_bits = 0;
    [[unroll]] for (uint j = 0; j < 4; ++j) {
        scale_bits |= (iqk_row_load_u32(block_offset + 4 * (4 * ib32 + j)) & 0x00010001) << (2 * j);
    }
    const uint scale = (scale_bits | (scale_bits >> 15)) & 0xFF;
    const u8vec4 indexes = unpack8((values >> (4 * (pos / 16))) & 0x0F0F0F0F);
    const uint table_offset = 16 * (scale & 1);
    const float d = iqk_row_scale(row_offset) * float(int(scale & 254) - 127);
    return d * vec4(float(kvalues_iq4_kss[indexes.x + table_offset]),
                    float(kvalues_iq4_kss[indexes.y + table_offset]),
                    float(kvalues_iq4_kss[indexes.z + table_offset]),
                    float(kvalues_iq4_kss[indexes.w + table_offset]));
#elif defined(DATA_A_IQ2_KS)
    const uint half_idx = element / 128;
    const uint group = (element % 128) / 32;
    const uint pos = element % 32;
    const uint extra = iqk_row_load_u16(block_offset) >> (4 * half_idx);
    const uint packed_scale = iqk_row_load_u8(block_offset + 2 + 2 * half_idx + group / 2);
    const uint scale = ((packed_scale >> (4 * (group % 2))) & 0x0F) |
                       (((extra >> (8 + group)) & 1) << 4);
    const uint values = iqk_row_load_u32(block_offset + 6 + 32 * half_idx + pos);
    const u8vec4 indexes = (unpack8(values) >> int8_t(2 * group)) & int8_t(3);
    const uint table_offset = 4 * ((extra >> group) & 1);
    const float d = iqk_row_scale(row_offset) * float(int(scale) - 16);
    return d * vec4(float(kvalues_iq2_ks[indexes.x + table_offset]),
                    float(kvalues_iq2_ks[indexes.y + table_offset]),
                    float(kvalues_iq2_ks[indexes.z + table_offset]),
                    float(kvalues_iq2_ks[indexes.w + table_offset]));
#elif defined(DATA_A_IQ2_KL)
    const uint ib64 = element / 64;
    const uint ib32_scale = element / 32;
    const uint pos64 = element % 64;
    const uint pos32 = pos64 % 32;
    const uint half_idx = pos64 / 32;
    const uint next_pos32 = pos32 + 2;
    const uint packed = iqk_row_load_u16(block_offset + 6 + 16 * ib64 + pos32 / 2);
    const uint high = iqk_row_load_u16(block_offset + 70 + pos32 / 2);
    const uint shift = 4 * half_idx;
    const uint high_shift = 2 * ib64 + half_idx;
    const uint index0 = ((packed >> shift) & 0x0F) | (((high >> high_shift) & 1) << 4);
    const uint index1 = (((packed >> 8) >> shift) & 0x0F) | ((((high >> 8) >> high_shift) & 1) << 4);
    const uint scale_low = (iqk_row_load_u8(block_offset + 2 + ib32_scale % 4) >> (4 * (ib32_scale / 4))) & 0x0F;
    const uint scale_high = (iqk_row_load_u16(block_offset) >> (2 * ib32_scale)) & 3;
    const float d = iqk_row_scale(row_offset) * float(int(scale_low | (scale_high << 4)) - 32);
    return d * vec4(float(kvalues_iq2_kl[2 * index0 + (pos32 & 1)]),
                    float(kvalues_iq2_kl[2 * index0 + ((pos32 + 1) & 1)]),
                    float(kvalues_iq2_kl[2 * index1 + (next_pos32 & 1)]),
                    float(kvalues_iq2_kl[2 * index1 + ((next_pos32 + 1) & 1)]));
#elif defined(DATA_A_IQ3_KS)
    const uint half_idx = element / 128;
    const uint group = (element % 128) / 32;
    const uint pos = element % 32;
    const uint extra = iqk_row_load_u16(block_offset) >> (4 * half_idx);
    const uint scale = ((iqk_row_load_u8(block_offset + 2 + group) >> (4 * half_idx)) & 0x0F) |
                       (((extra >> group) & 1) << 4);
    const uint low = iqk_row_load_u32(block_offset + 6 + 32 * half_idx + pos);
    const uint high = iqk_row_load_u32(block_offset + 70 + pos);
    const uint extra_bit = ((extra >> (8 + group)) & 1) << 3;
    const u8vec4 indexes = ((unpack8(low) >> int8_t(2 * group)) & int8_t(3)) |
                           (((unpack8(high) >> int8_t(4 * half_idx + group)) & int8_t(1)) << int8_t(2)) |
                           u8vec4(extra_bit);
    const float d = iqk_row_scale(row_offset) * float(int(scale) - 16);
    return d * vec4(float(kvalues_iq3_ks[indexes.x]),
                    float(kvalues_iq3_ks[indexes.y]),
                    float(kvalues_iq3_ks[indexes.z]),
                    float(kvalues_iq3_ks[indexes.w]));
#elif defined(DATA_A_IQ4_KS)
    const uint ib32 = element / 32;
    const uint pos = element % 32;
    const uint scale = iqk_row_load_u8(block_offset + ib32);
    const uint values = iqk_row_load_u32(block_offset + 8 + 16 * ib32 + pos % 16);
    const u8vec4 indexes = (unpack8(values) >> int8_t(4 * (pos / 16))) & int8_t(0x0F);
    const uint table_offset = 16 * (scale & 1);
    const float d = iqk_row_scale(row_offset) * float(int(scale & 254) - 127);
    return d * vec4(float(kvalues_iq4_ks[indexes.x + table_offset]),
                    float(kvalues_iq4_ks[indexes.y + table_offset]),
                    float(kvalues_iq4_ks[indexes.z + table_offset]),
                    float(kvalues_iq4_ks[indexes.w + table_offset]));
#else
    const uint ib64 = element / 64;
    const uint pos64 = element % 64;
    const uint pos32 = pos64 % 32;
    const uint half_idx = pos64 / 32;
    const uint scale = iqk_row_load_u8(block_offset + 2 * ib64 + half_idx);
    const uint values = iqk_row_load_u32(block_offset + 8 + 32 * ib64 + pos32);
    const uint high = iqk_row_load_u32(block_offset + 136 + pos32);
    const uint value_shift = 4 * half_idx;
    const uint high_shift = 2 * ib64 + half_idx;
    const uint table_offset = 32 * (scale & 1);
    const u8vec4 indexes = ((unpack8(values) >> int8_t(value_shift)) & int8_t(0x0F)) |
                           (((unpack8(high) >> int8_t(high_shift)) & int8_t(1)) << int8_t(4));
    const float d = iqk_row_scale(row_offset) * float(int(scale & 254) - 127);
    return d * vec4(float(kvalues_iq5_ks[indexes.x + table_offset]),
                    float(kvalues_iq5_ks[indexes.y + table_offset]),
                    float(kvalues_iq5_ks[indexes.z + table_offset]),
                    float(kvalues_iq5_ks[indexes.w + table_offset]));
#endif
}
#endif
#endif

#if defined(DATA_A_TQ1_0)
float tq1_0_val(uint ib, uint e, uint a_offset) {
    const uint bidx = tq1_0_byte_of(e);
    const uint qbyte = uint(bidx < 48u ? data_a[a_offset + ib].qs[bidx]
                                       : data_a[a_offset + ib].qh[bidx - 48u]);
    return float(tq1_0_trit(qbyte, tq1_0_digit_of(e))) - 1.0;
}
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(tq1_0_val(ib, iqs, a_offset), tq1_0_val(ib, iqs + 1u, a_offset));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(float(data_a[a_offset + ib].d), 0);
}
#endif

#if defined(DATA_A_TQ2_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    // elem e -> byte qs[(e/128)*32 + e%32], bits 2*((e%128)/32); w = q - 1 (d applied via get_dm)
    const uint qsi   = (iqs / 128) * 32 + (iqs % 32);  // iqs even -> qsi, qsi+1 in same group/level
    const uint shift = 2 * ((iqs % 128) / 32);

    const uvec2 qs = uvec2(data_a[a_offset + ib].qs[qsi], data_a[a_offset + ib].qs[qsi + 1]);
    return vec2((qs >> shift) & 3) - 1.0;
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(float(data_a[a_offset + ib].d), 0);
}
#endif

#if defined(DATA_A_Q3_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 64;                     // 0,1
    const uint qsi = n * 32 + (iqs % 16) * 2;    // 0,2,4..62
    const uint hmi =          (iqs % 16) * 2;    // 0,2,4..30
    const uint j = (iqs % 64) / 4;               // 0..3
    const uint is = iqs / 8;                     // 0..15
    const uint halfsplit = ((iqs % 64) / 16);    // 0,1,2,3
    const uint qsshift = halfsplit * 2;          // 0,2,4,6
    const uint m = 1 << (4 * n + halfsplit);     // 1,2,4,8,16,32,64,128

    const int8_t us = int8_t(((data_a[a_offset + ib].scales[is % 8] >> (4 * int(is / 8))) & 0xF)
                          | (((data_a[a_offset + ib].scales[8 + (is % 4)] >> (2 * int(is / 4))) & 3) << 4));
    const float dl = float(data_a[a_offset + ib].d) * float(us - 32);

    return vec2(dl * float(int8_t((data_a[a_offset + ib].qs[qsi    ] >> qsshift) & 3) - (((data_a[a_offset + ib].hmask[hmi    ] & m) != 0) ? 0 : 4)),
                dl * float(int8_t((data_a[a_offset + ib].qs[qsi + 1] >> qsshift) & 3) - (((data_a[a_offset + ib].hmask[hmi + 1] & m) != 0) ? 0 : 4)));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q4_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 32;                   // 0,1,2,3
    const uint b = (iqs % 32) / 16;            // 0,1
    const uint is = 2 * n + b;                 // 0..7
    const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126

    const vec2 loadd = vec2(data_a[a_offset + ib].dm);

    const uint scidx0 = (is < 4) ? is : (is + 4);
    const uint scidx1 = (is < 4) ? is : (is - 4);
    const uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint scidxshift1 = (is < 4) ? 0 : 2;
    const uint mbidx0 = is + 4;
    const uint mbidx1 = (is < 4) ? is + 4 : is;
    const uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
    const uint mbidxshift0 = (is < 4) ? 0 : 4;
    const uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint mbidxshift1 = (is < 4) ? 0 : 2;

    const uint8_t sc = uint8_t((data_a[a_offset + ib].scales[scidx0] & 0xF) | ((data_a[a_offset + ib].scales[scidx1] & scidxmask1) >> scidxshift1));
    const uint8_t mbyte = uint8_t((data_a[a_offset + ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0 | ((data_a[a_offset + ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

    const float d = loadd.x * sc;
    const float m = -loadd.y * mbyte;

    return vec2(fma(d, float((data_a[a_offset + ib].qs[qsi    ] >> (b * 4)) & 0xF), m),
                fma(d, float((data_a[a_offset + ib].qs[qsi + 1] >> (b * 4)) & 0xF), m));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q5_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 32;                   // 0,1,2,3
    const uint b = (iqs % 32) / 16;            // 0,1
    const uint is = 2 * n + b;                 // 0..7
    const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126
    const uint qhi = (iqs % 16) * 2;           // 0,2,4..30

    const uint8_t hm = uint8_t(1 << (iqs / 16));

    const vec2 loadd = vec2(data_a[a_offset + ib].dm);

    const uint scidx0 = (is < 4) ? is : (is + 4);
    const uint scidx1 = (is < 4) ? is : (is - 4);
    const uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint scidxshift1 = (is < 4) ? 0 : 2;
    const uint mbidx0 = is + 4;
    const uint mbidx1 = (is < 4) ? is + 4 : is;
    const uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
    const uint mbidxshift0 = (is < 4) ? 0 : 4;
    const uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint mbidxshift1 = (is < 4) ? 0 : 2;

    const uint8_t sc    = uint8_t((data_a[a_offset + ib].scales[scidx0] & 0xF)                         | ((data_a[a_offset + ib].scales[scidx1] & scidxmask1) >> scidxshift1));
    const uint8_t mbyte = uint8_t(((data_a[a_offset + ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0) | ((data_a[a_offset + ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

    const float d = loadd.x * sc;
    const float m = -loadd.y * mbyte;

    return vec2(fma(d, float((data_a[a_offset + ib].qs[qsi    ] >> (b * 4)) & 0xF) + float((data_a[a_offset + ib].qh[qhi    ] & hm) != 0 ? 16 : 0), m),
                fma(d, float((data_a[a_offset + ib].qs[qsi + 1] >> (b * 4)) & 0xF) + float((data_a[a_offset + ib].qh[qhi + 1] & hm) != 0 ? 16 : 0), m));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q6_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 64;                    // 0,1
    const uint b = (iqs % 64) / 32;             // 0,1
    const uint is_b = (iqs % 16) / 8;           // 0,1
    const uint qhshift = ((iqs % 64) / 16) * 2; // 0,2,4,6
    const uint is = 8 * n + qhshift + is_b;     // 0..15
    const uint qsi = n * 64 + (iqs % 32) * 2;   // 0,2,4..126
    const uint qhi = n * 32 + (iqs % 16) * 2;   // 0,2,4..62

    const float dscale = float(data_a[a_offset + ib].d) * float(data_a[a_offset + ib].scales[is]);

    return vec2(dscale * float(int8_t(((data_a[a_offset + ib].ql[qsi    ] >> (b * 4)) & 0xF) | (((data_a[a_offset + ib].qh[qhi    ] >> qhshift) & 3) << 4)) - 32),
                dscale * float(int8_t(((data_a[a_offset + ib].ql[qsi + 1] >> (b * 4)) & 0xF) | (((data_a[a_offset + ib].qh[qhi + 1] >> qhshift) & 3) << 4)) - 32));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif
