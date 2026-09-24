#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"
#include "mmq_iqk.glsl"

// Each iqs value maps to a 32-bit integer

#if defined(DATA_A_Q2_0)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint block_idx = ib / 2;
    const uint byte_idx = (ib & 1u) * 8u + iqs;
    const uint bits = uint(data_a[block_idx].qs[byte_idx]);
    buf_a[buf_ib].qs[iqs] = pack32(i8vec4(
        int8_t(bits & 3u),
        int8_t((bits >> 2u) & 3u),
        int8_t((bits >> 4u) & 3u),
        int8_t(bits >> 6u)));

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE(data_a[block_idx].d);
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 8; ++iqs) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; ++iqs) {
        q_sum += dotPacked4x8EXT(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_a[ib_a].dm) * (float(q_sum) * float(cache_b.ds.x) - float(cache_b.ds.y)));
}
#endif

#if defined(DATA_A_Q4_0) || defined(DATA_A_Q4_1)
// 2-byte loads for Q4_0 blocks (18 bytes)
// 4-byte loads for Q4_1 blocks (20 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
#ifdef DATA_A_Q4_0
    buf_a[buf_ib].qs[iqs] = pack32(u16vec2(data_a_packed16[ib].qs[iqs * 2],
                                           data_a_packed16[ib].qs[iqs * 2 + 1]));

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
    }
#else // DATA_A_Q4_1
    buf_a[buf_ib].qs[iqs] = data_a_packed32[ib].qs[iqs];

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPEV2(data_a_packed32[ib].dm);
    }
#endif
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        const uint32_t vui = cache_a[ib_a].qs[iqs];
        const i32vec2 qs_a = i32vec2( vui       & 0x0F0F0F0F,
                                     (vui >> 4) & 0x0F0F0F0F);

        const int32_t qs_b0 = cache_b.qs[iqs];
        const int32_t qs_b1 = cache_b.qs[iqs + 4];

        q_sum += dotPacked4x8EXT(qs_a.x, qs_b0);
        q_sum += dotPacked4x8EXT(qs_a.y, qs_b1);
    }

#ifdef DATA_A_Q4_0
    return ACC_TYPE(float(cache_a[ib_a].dm) * (float(q_sum) * float(cache_b.ds.x) - 8.0 * float(cache_b.ds.y)));
#else // DATA_A_Q4_1
    return ACC_TYPE(float(q_sum) * float(cache_a[ib_a].dm.x) * float(cache_b.ds.x) + float(cache_a[ib_a].dm.y) * float(cache_b.ds.y));
#endif
}
#endif

#if defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1)
// 2-byte loads for Q5_0 blocks (22 bytes)
// 4-byte loads for Q5_1 blocks (24 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
#ifdef DATA_A_Q5_0
    buf_a[buf_ib].qs[iqs] = pack32(u16vec2(data_a_packed16[ib].qs[iqs * 2],
                                           data_a_packed16[ib].qs[iqs * 2 + 1]));

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
        buf_a[buf_ib].qh = pack32(u16vec2(data_a_packed16[ib].qh[0], data_a_packed16[ib].qh[1]));
    }
#else // DATA_A_Q5_1
    buf_a[buf_ib].qs[iqs] = data_a_packed32[ib].qs[iqs];

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPEV2(data_a_packed32[ib].dm);
        buf_a[buf_ib].qh = data_a_packed32[ib].qh;
    }
#endif
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;
    cache_a[reg_ib].qh = buf_a[buf_ib].qh;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        const uint32_t vui = cache_a[ib_a].qs[iqs];
        const int32_t qh = int32_t(cache_a[ib_a].qh >> (4 * iqs));
        const int32_t qs_a0 = int32_t(vui & 0x0F0F0F0F)
                         | ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)
        const int32_t qs_a1 = int32_t((vui >> 4) & 0x0F0F0F0F)
                         | (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)

        const int32_t qs_b0 = cache_b.qs[iqs];
        const int32_t qs_b1 = cache_b.qs[iqs + 4];

        q_sum += dotPacked4x8EXT(qs_a0, qs_b0);
        q_sum += dotPacked4x8EXT(qs_a1, qs_b1);
    }

#ifdef DATA_A_Q5_0
    return ACC_TYPE(float(cache_a[ib_a].dm) * (float(q_sum) * float(cache_b.ds.x) - 16.0 * float(cache_b.ds.y)));
#else // DATA_A_Q5_1
    return ACC_TYPE(float(q_sum) * float(cache_a[ib_a].dm.x) * float(cache_b.ds.x) + float(cache_a[ib_a].dm.y) * float(cache_b.ds.y));
#endif
}
#endif

#if defined(DATA_A_Q8_0)
// 2-byte loads for Q8_0 blocks (34 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    buf_a[buf_ib].qs[iqs] = pack32(i16vec2(data_a_packed16[ib].qs[iqs * 2],
                                           data_a_packed16[ib].qs[iqs * 2 + 1]));

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];
        const int32_t qs_b = cache_b.qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, qs_b);
    }

    return ACC_TYPE(float(q_sum) * float(cache_a[ib_a].dm) * float(cache_b.ds.x));
}
#endif

#if defined(DATA_A_MXFP4)
// 1-byte loads for mxfp4 blocks (17 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint32_t qs = pack32(u8vec4(data_a[ib].qs[iqs * 4    ],
                                      data_a[ib].qs[iqs * 4 + 1],
                                      data_a[ib].qs[iqs * 4 + 2],
                                      data_a[ib].qs[iqs * 4 + 3]));

    const u8vec4 i_a0 = unpack8( qs       & 0x0F0F0F0F);
    const u8vec4 i_a1 = unpack8((qs >> 4) & 0x0F0F0F0F);

    buf_a[buf_ib].qs[iqs    ] = pack32(i8vec4(kvalues_mxfp4[i_a0.x], kvalues_mxfp4[i_a0.y], kvalues_mxfp4[i_a0.z], kvalues_mxfp4[i_a0.w]));
    buf_a[buf_ib].qs[iqs + 4] = pack32(i8vec4(kvalues_mxfp4[i_a1.x], kvalues_mxfp4[i_a1.y], kvalues_mxfp4[i_a1.z], kvalues_mxfp4[i_a1.w]));

    if (iqs == 0) {
        buf_a[buf_ib].d = FLOAT_TYPE(e8m0_to_fp32(data_a[ib].e) * 0.5);
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d = buf_a[buf_ib].d;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_a[ib_a].d) * float(cache_b.ds.x) * float(q_sum));
}
#endif

#if defined(DATA_A_IQ4_XS)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint32_t vui = data_a_packed32[ib_k].qs[4 * ib32 + iqs];
    const i32vec2 qs = iq4nl_to_i8x8(vui);

    buf_a[buf_ib].qs[iqs    ] = qs.x;
    buf_a[buf_ib].qs[iqs + 4] = qs.y;

    if (iqs == 0) {
        const uint sl = (data_a_packed32[ib_k].scales_l >> (4 * ib32)) & 0xF;
        const uint sh = (data_a_packed32[ib_k].scales_h >> (2 * ib32)) & 3;
        buf_a[buf_ib].d = FLOAT_TYPE(float(data_a[ib_k].d) * float(int(sl | (sh << 4)) - 32));
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d = buf_a[buf_ib].d;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        q_sum += dotPacked4x8EXT(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_a[ib_a].d) * float(cache_b.ds.x) * float(q_sum));
}
#endif

#if defined(DATA_A_IQ2_K)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint8_t ib32 = uint8_t(ib) & uint8_t(7);
    const uint8_t iqs8 = uint8_t(iqs);
    const uint8_t half_idx = iqs8 >> 2;
    const uint8_t qs_idx = (bool(ib32 & uint8_t(4)) ? uint8_t(8) : uint8_t(0)) + iqs8;
    const uint8_t ib32_shift = ib32 << 1;
    const uint8_t shift = ib32_shift & uint8_t(6);
    const bool is_hi_table = bool((data_a[ib_k].extra >> (ib32_shift + half_idx)) & uint16_t(1));

    buf_a[buf_ib].qs[iqs8] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx] >> shift) & 0x03030303, is_hi_table);

    if (iqs8 == uint8_t(0)) {
        const uint8_t scales = data_a[ib_k].scales[ib32];
        const vec2 block_scales = vec2(int32_t(scales & uint8_t(0x0F)) - 8, int32_t(scales >> 4) - 8);
        buf_a[buf_ib].d_scales = FLOAT_TYPEV2(float(data_a[ib_k].d) * block_scales);
    }
}
#endif

#if defined(DATA_A_IQ3_K)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint8_t ib32 = uint8_t(ib) & uint8_t(7);
    const uint8_t iqs8 = uint8_t(iqs);
    const uint8_t half_idx = iqs8 >> 2;
    const uint8_t qh_idx = iqs8 << 1;
    const uint8_t q_idx = (bool(ib32 & uint8_t(4)) ? uint8_t(16) : uint8_t(0)) + qh_idx;
    const bool is_hi_table = bool((data_a[ib_k].extra >> ((ib32 << 1) + half_idx)) & uint16_t(1));
    const uint ql = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx], data_a_packed16[ib_k].qs[q_idx + 1]));
    const uint qh = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_idx], data_a_packed16[ib_k].qh[qh_idx + 1]));

    buf_a[buf_ib].qs[iqs8] = unpack_iq3_k(ql, qh, ib32, is_hi_table);

    if (iqs8 == uint8_t(0)) {
        const uint8_t scales_l = data_a[ib_k].scales_l[ib32];
        const uint16_t scales_h = data_a[ib_k].scales_h >> (ib32 << 1);
        const float scale0 = float(2 * (scales_l & uint8_t(0x0F)) + 1) * (bool(scales_h & uint16_t(1)) ? -1.0 : 1.0);
        const float scale1 = float(2 * (scales_l >> 4) + 1) * (bool(scales_h & uint16_t(2)) ? -1.0 : 1.0);
        buf_a[buf_ib].d_scales = FLOAT_TYPEV2(float(data_a[ib_k].d) * vec2(scale0, scale1));
    }
}
#endif

#if defined(DATA_A_IQ4_K)
int32_t unpack_iq4_k(uint32_t values, uint shift, uint table_offset) {
    const u8vec4 indexes = unpack8((values >> shift) & 0x0F0F0F0F);
    return pack32(i8vec4(kvalues_iq4_k[indexes.x + table_offset],
                         kvalues_iq4_k[indexes.y + table_offset],
                         kvalues_iq4_k[indexes.z + table_offset],
                         kvalues_iq4_k[indexes.w + table_offset]));
}

void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint half_idx = iqs / 4;
    const uint qs_idx = 4 * ib32 + iqs % 4;
    const uint table_offset = 16 * ((uint(data_a[ib_k].extra) >> (2 * ib32 + half_idx)) & 1);

    buf_a[buf_ib].qs[iqs] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx], 4 * half_idx, table_offset);

    if (iqs == 0) {
        const uint scales_h = uint(data_a[ib_k].scales_h[ib32 / 2]) >> (4 * (ib32 % 2));
        const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
        const int32_t scale0 = int32_t((scales_l & 0x0F) | ((scales_h << 4) & 0x30)) - 32;
        const int32_t scale1 = int32_t((scales_l >> 4) | ((scales_h << 2) & 0x30)) - 32;
        buf_a[buf_ib].d_scales = FLOAT_TYPEV2(float(data_a[ib_k].d) * vec2(scale0, scale1));
    }
}
#endif

#if defined(DATA_A_IQ5_K)
int32_t unpack_iq5_k(uint32_t values, uint32_t qh, uint shift, uint table_offset) {
    const uint32_t low = (values >> (4 * (shift & 1))) & 0x0F0F0F0Fu;
    const uint32_t high = ((qh >> shift) & 0x01010101u) << 4u;
    const u8vec4 indexes = unpack8(low | high);
    return pack32(i8vec4(kvalues_iq5_k[indexes.x + table_offset],
                         kvalues_iq5_k[indexes.y + table_offset],
                         kvalues_iq5_k[indexes.z + table_offset],
                         kvalues_iq5_k[indexes.w + table_offset]));
}

void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint half_idx = iqs / 4;
    const uint qs_idx = 8 * (ib32 / 2) + iqs;
    const uint table_offset = 32 * ((uint(data_a[ib_k].extra) >> (2 * ib32 + half_idx)) & 1);

    buf_a[buf_ib].qs[iqs] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx], data_a_packed32[ib_k].qh[iqs], ib32, table_offset);

    if (iqs == 0) {
        const uint scales_h = uint(data_a[ib_k].scales_h[ib32 / 2]) >> (4 * (ib32 % 2));
        const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
        const int32_t scale0 = int32_t((scales_l & 0x0F) | ((scales_h << 4) & 0x30)) - 32;
        const int32_t scale1 = int32_t((scales_l >> 4) | ((scales_h << 2) & 0x30)) - 32;
        buf_a[buf_ib].d_scales = FLOAT_TYPEV2(float(data_a[ib_k].d) * vec2(scale0, scale1));
    }
}
#endif

#if defined(DATA_A_IQ6_K)
int32_t unpack_iq6_k(uint32_t values, uint32_t qh, uint shift, uint table_offset) {
    const uint32_t low = (values >> (4 * ((shift / 2) & 1))) & 0x0F0F0F0Fu;
    const uint32_t high = ((qh >> shift) & 0x03030303u) << 4u;
    const u8vec4 indexes = unpack8(low | high);
    return pack32(i8vec4(kvalues_iq6_k[indexes.x + table_offset],
                         kvalues_iq6_k[indexes.y + table_offset],
                         kvalues_iq6_k[indexes.z + table_offset],
                         kvalues_iq6_k[indexes.w + table_offset]));
}

void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint half_idx = iqs / 4;
    const uint qs_idx = 8 * (ib32 / 2) + iqs;
    const uint qh_idx = 8 * (ib32 / 4) + iqs;
    const uint shift = 2 * (ib32 % 4);
    const uint table_offset = 64 * ((uint(data_a[ib_k].extra) >> (2 * ib32 + half_idx)) & 1);

    buf_a[buf_ib].qs[iqs] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx], data_a_packed32[ib_k].qh[qh_idx], shift, table_offset);

    if (iqs == 0) {
        const float scale0 = float(data_a[ib_k].scales[2 * ib32]);
        const float scale1 = float(data_a[ib_k].scales[2 * ib32 + 1]);
        buf_a[buf_ib].d_scales = FLOAT_TYPEV2(float(data_a[ib_k].d) * vec2(scale0, scale1));
    }
}
#endif

#if defined(DATA_A_IQK_ROW)
#if defined(DATA_A_IQ1_KT) || defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT) || defined(DATA_A_IQ4_KT)
void mmq_iqkt_take4(inout uint state, uint count, out int value0, out int value1, out int value2, out int value3) {
    [[unroll]] for (uint j = 0; j < count; ++j) {
        iqkt_next(state);
    }
    value0 = iqkt_next(state);
    value1 = iqkt_next(state);
    value2 = iqkt_next(state);
    value3 = iqkt_next(state);
}
#endif

void mmq_iqks_pack8(uint block_offset, uint element, out uint value0, out uint value1, out float d_scale) {
    d_scale = 0.0;
#if defined(DATA_A_IQ1_KT) || defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT) || defined(DATA_A_IQ4_KT)
    const uint group8 = element / 8;
    const uint pos8 = element % 8;
    const uint ib32 = element / 32;
    const uint group8_next = group8 + 2;
    uint state0;
    uint state1;

#if defined(DATA_A_IQ1_KT)
    const uint8_t sh = iqks_load_u8(block_offset + ib32);
    const uint8_t x0 = iqks_load_u8(block_offset + 8 + group8);
    const uint8_t x1 = iqks_load_u8(block_offset + 8 + group8_next);
    const uint8_t y0 = (iqks_load_u8(block_offset + 40 + group8 % 16) >> (4 * (group8 / 16))) & uint8_t(0x0F);
    const uint8_t y1 = (iqks_load_u8(block_offset + 40 + group8_next % 16) >> (4 * (group8_next / 16))) & uint8_t(0x0F);
    const bool z0 = bool(sh & (uint8_t(1) << (4 + group8 % 4)));
    const bool z1 = bool(sh & (uint8_t(1) << (4 + group8_next % 4)));
    state0 = pack32(u8vec4(x0, y0 | (z0 ? uint8_t(2) : uint8_t(1)) << 4, uint8_t(0), uint8_t(0)));
    state1 = pack32(u8vec4(x1, y1 | (z1 ? uint8_t(2) : uint8_t(1)) << 4, uint8_t(0), uint8_t(0)));
    if (element % 32 == 0) {
        d_scale = float(kvalues_iqkt_scale[sh & 0x0F]);
    }
#elif defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT)
    state0 = uint(iqks_load_u16(block_offset + 4 + 2 * group8)) + 4096u;
    state1 = uint(iqks_load_u16(block_offset + 4 + 2 * group8_next)) + 4096u;
    if (element % 32 == 0) {
        const uint8_t scale = (iqks_load_u8(block_offset + ib32 % 4) >> (4 * (ib32 / 4))) & uint8_t(0x0F);
#if defined(DATA_A_IQ2_KT)
        d_scale = float(kvalues_iqkt_scale[scale]);
#else
        d_scale = float(scale);
#endif
    }
#else
    const uint header = iqks_load_u32(block_offset + 4 * ib32);
    const uint seed_index0 = 2 * group8 + pos8 / 4;
    const uint seed_index1 = 2 * group8_next + pos8 / 4;
    const uint8_t x0 = iqks_load_u8(block_offset + 32 + seed_index0);
    const uint8_t x1 = iqks_load_u8(block_offset + 32 + seed_index1);
    const uint8_t y0 = (iqks_load_u8(block_offset + 96 + seed_index0 % 32) >> (4 * (seed_index0 / 32))) & uint8_t(0x0F);
    const uint8_t y1 = (iqks_load_u8(block_offset + 96 + seed_index1 % 32) >> (4 * (seed_index1 / 32))) & uint8_t(0x0F);
    const uint8_t z0 = uint8_t((header >> (8 + 3 * (seed_index0 % 8))) & 7u);
    const uint8_t z1 = uint8_t((header >> (8 + 3 * (seed_index1 % 8))) & 7u);
    const bool w = bool(header & 1u);
    const uint8_t high0 = z0 + uint8_t(1) + (w ? uint8_t(8) : uint8_t(0));
    const uint8_t high1 = z1 + uint8_t(1) + (w ? uint8_t(8) : uint8_t(0));
    state0 = pack32(u8vec4(x0, y0 | ((high0 & uint8_t(0x0F)) << 4), high0 >> 4, uint8_t(0)));
    state1 = pack32(u8vec4(x1, y1 | ((high1 & uint8_t(0x0F)) << 4), high1 >> 4, uint8_t(0)));
    if (element % 32 == 0) {
        d_scale = float(int((header & 0xFF) >> 1) - 64);
    }
#endif

    const uint count =
#if defined(DATA_A_IQ4_KT)
        pos8 % 4;
#else
        pos8;
#endif
    int v00;
    int v01;
    int v02;
    int v03;
    int v10;
    int v11;
    int v12;
    int v13;
    mmq_iqkt_take4(state0, count, v00, v01, v02, v03);
    mmq_iqkt_take4(state1, count, v10, v11, v12, v13);

#if defined(DATA_A_IQ3_KT)
    v00 = abs(v00);
    v01 = abs(v01);
    v02 = abs(v02);
    v03 = abs(v03);
    v10 = abs(v10);
    v11 = abs(v11);
    v12 = abs(v12);
    v13 = abs(v13);
    const u8vec4 signs0 = unpack8((iqks_load_u32(block_offset + 68 + element % 32) >> ib32) & 0x01010101u);
    const u8vec4 signs1 = unpack8((iqks_load_u32(block_offset + 68 + (element + 16) % 32) >> ib32) & 0x01010101u);
    if (signs0.x != uint8_t(0)) v00 = -v00;
    if (signs0.y != uint8_t(0)) v01 = -v01;
    if (signs0.z != uint8_t(0)) v02 = -v02;
    if (signs0.w != uint8_t(0)) v03 = -v03;
    if (signs1.x != uint8_t(0)) v10 = -v10;
    if (signs1.y != uint8_t(0)) v11 = -v11;
    if (signs1.z != uint8_t(0)) v12 = -v12;
    if (signs1.w != uint8_t(0)) v13 = -v13;
#endif

    value0 = pack32(i8vec4(int8_t(v00), int8_t(v01), int8_t(v02), int8_t(v03)));
    value1 = pack32(i8vec4(int8_t(v10), int8_t(v11), int8_t(v12), int8_t(v13)));
#elif defined(DATA_A_IQ4_KSS)
    const uint ib32 = element / 32;
    const uint pos = element % 32;
    const uint word_offset = block_offset + 4 * (4 * ib32 + (pos % 16) / 4);
    uint values = iqks_load_u32(word_offset) & 0xFFFEFFFE;
    values ^= values >> 1;

    uint scale_bits = 0;
    [[unroll]] for (uint j = 0; j < 4; ++j) {
        scale_bits |= (iqks_load_u32(block_offset + 4 * (4 * ib32 + j)) & 0x00010001) << (2 * j);
    }
    const uint8_t scale = uint8_t(scale_bits | (scale_bits >> 15));
    if (element % 32 == 0) {
        d_scale = float(int(scale & uint8_t(254)) - 127);
    }
    const uint table_offset = bool(scale & uint8_t(1)) ? 16u : 0u;
    const u8vec4 indexes0 = unpack8(values & 0x0F0F0F0Fu);
    const u8vec4 indexes1 = unpack8((values >> 4) & 0x0F0F0F0Fu);

    value0 = pack32(i8vec4(kvalues_iq4_k[indexes0.x + table_offset],
                           kvalues_iq4_k[indexes0.y + table_offset],
                           kvalues_iq4_k[indexes0.z + table_offset],
                           kvalues_iq4_k[indexes0.w + table_offset]));
    value1 = pack32(i8vec4(kvalues_iq4_k[indexes1.x + table_offset],
                           kvalues_iq4_k[indexes1.y + table_offset],
                           kvalues_iq4_k[indexes1.z + table_offset],
                           kvalues_iq4_k[indexes1.w + table_offset]));
#elif defined(DATA_A_IQ4_KS)
    const uint ib32 = element / 32;
    const uint pos = element % 32;
    const uint8_t scale = iqks_load_u8(block_offset + ib32);
    const uint values = iqks_load_u32(block_offset + 8 + 16 * ib32 + pos % 16);
    const uint table_offset = bool(scale & uint8_t(1)) ? 16u : 0u;
    const u8vec4 unpacked_values = unpack8(values);
    const u8vec4 indexes0 = unpacked_values & int8_t(0x0F);
    const u8vec4 indexes1 = (unpacked_values >> int8_t(4)) & int8_t(0x0F);
    if (element % 32 == 0) {
        d_scale = float(int(scale & uint8_t(254)) - 127);
    }

    value0 = pack32(i8vec4(kvalues_iq4_k[indexes0.x + table_offset],
                           kvalues_iq4_k[indexes0.y + table_offset],
                           kvalues_iq4_k[indexes0.z + table_offset],
                           kvalues_iq4_k[indexes0.w + table_offset]));
    value1 = pack32(i8vec4(kvalues_iq4_k[indexes1.x + table_offset],
                           kvalues_iq4_k[indexes1.y + table_offset],
                           kvalues_iq4_k[indexes1.z + table_offset],
                           kvalues_iq4_k[indexes1.w + table_offset]));
#elif defined(DATA_A_IQ2_KS)
    const uint half_idx = element / 128;
    const uint group = (element % 128) / 32;
    const uint16_t extra = iqks_load_u16(block_offset) >> (4 * half_idx);
    const bool is_hi_table = bool((extra >> group) & 1);
    const uint values0 = iqks_load_u32(block_offset + 6 + 32 * half_idx + element % 32);
    const uint values1 = iqks_load_u32(block_offset + 6 + 32 * half_idx + element % 32 + 16);
    if (element % 32 == 0) {
        const uint8_t packed_scale = iqks_load_u8(block_offset + 2 + 2 * half_idx + group / 2);
        const uint8_t scale = ((packed_scale >> (4 * (group % 2))) & uint8_t(0x0F)) |
                              (bool(extra & (uint16_t(1) << (8 + group))) ? uint8_t(16) : uint8_t(0));
        d_scale = float(int(scale) - 16);
    }

    value0 = uint(unpack_iq2_k((values0 >> (2 * group)) & 0x03030303u, is_hi_table));
    value1 = uint(unpack_iq2_k((values1 >> (2 * group)) & 0x03030303u, is_hi_table));
#elif defined(DATA_A_IQ3_KS)
    const uint half_idx = element / 128;
    const uint8_t shift_h = uint8_t(element) >> 5;
    const uint8_t group = shift_h & uint8_t(3);
    const uint16_t extra = iqks_load_u16(block_offset) >> (4 * half_idx);
    const uint low0 = iqks_load_u32(block_offset + 6 + 32 * half_idx + element % 32);
    const uint low1 = iqks_load_u32(block_offset + 6 + 32 * half_idx + element % 32 + 16);
    const uint high0 = iqks_load_u32(block_offset + 70 + element % 32);
    const uint high1 = iqks_load_u32(block_offset + 70 + element % 32 + 16);
    const bool is_hi_table = bool((extra >> (8 + group)) & 1);
    if (element % 32 == 0) {
        const uint8_t scale = ((iqks_load_u8(block_offset + 2 + group) >> (4 * half_idx)) & uint8_t(0x0F)) |
                              (bool(extra & (uint16_t(1) << group)) ? uint8_t(16) : uint8_t(0));
        d_scale = float(int(scale) - 16);
    }

    value0 = uint(unpack_iq3_k(low0, high0, shift_h, is_hi_table));
    value1 = uint(unpack_iq3_k(low1, high1, shift_h, is_hi_table));
#elif defined(DATA_A_IQ5_KS)
    const uint ib64 = element / 64;
    const uint pos64 = element % 64;
    const uint pos32 = pos64 % 32;
    const uint half_idx = pos64 / 32;
    const uint8_t scale = iqks_load_u8(block_offset + 2 * ib64 + half_idx);
    const uint values0 = iqks_load_u32(block_offset + 8 + 32 * ib64 + pos32);
    const uint values1 = iqks_load_u32(block_offset + 8 + 32 * ib64 + pos32 + 16);
    const uint high0 = iqks_load_u32(block_offset + 136 + pos32);
    const uint high1 = iqks_load_u32(block_offset + 136 + pos32 + 16);
    const uint value_shift = 4 * half_idx;
    const uint high_shift = 2 * ib64 + half_idx;
    const uint table_offset = bool(scale & uint8_t(1)) ? 32u : 0u;
    const u8vec4 indexes0 = ((unpack8(values0) >> int8_t(value_shift)) & int8_t(0x0F)) |
                            (((unpack8(high0) >> int8_t(high_shift)) & int8_t(1)) << int8_t(4));
    const u8vec4 indexes1 = ((unpack8(values1) >> int8_t(value_shift)) & int8_t(0x0F)) |
                            (((unpack8(high1) >> int8_t(high_shift)) & int8_t(1)) << int8_t(4));
    if (element % 32 == 0) {
        d_scale = float(int(scale & uint8_t(254)) - 127);
    }

    value0 = pack32(i8vec4(kvalues_iq5_k[indexes0.x + table_offset],
                           kvalues_iq5_k[indexes0.y + table_offset],
                           kvalues_iq5_k[indexes0.z + table_offset],
                           kvalues_iq5_k[indexes0.w + table_offset]));
    value1 = pack32(i8vec4(kvalues_iq5_k[indexes1.x + table_offset],
                           kvalues_iq5_k[indexes1.y + table_offset],
                           kvalues_iq5_k[indexes1.z + table_offset],
                           kvalues_iq5_k[indexes1.w + table_offset]));
#elif defined(DATA_A_IQ2_KL)
    const uint ib64 = element / 64;
    const uint ib32_scale = element / 32;
    const uint pos64 = element % 64;
    const uint pos32 = pos64 % 32;
    const uint half_idx = pos64 / 32;
    const uint next_pos32 = pos32 + 16;
    const uint16_t packed0 = iqks_load_u16(block_offset + 6 + 16 * ib64 + pos32 / 2);
    const uint16_t packed1 = iqks_load_u16(block_offset + 6 + 16 * ib64 + next_pos32 / 2);
    const uint16_t high0 = iqks_load_u16(block_offset + 70 + pos32 / 2);
    const uint16_t high1 = iqks_load_u16(block_offset + 70 + next_pos32 / 2);
    const uint shift = 4 * half_idx;
    const uint high_shift = 2 * ib64 + half_idx;
    const uint8_t index00 = uint8_t(((packed0 >> shift) & uint16_t(0x0F)) | (((high0 >> high_shift) & uint16_t(1)) << 4));
    const uint8_t index01 = uint8_t((((packed0 >> 8) >> shift) & uint16_t(0x0F)) | ((((high0 >> 8) >> high_shift) & uint16_t(1)) << 4));
    const uint8_t index10 = uint8_t(((packed1 >> shift) & uint16_t(0x0F)) | (((high1 >> high_shift) & uint16_t(1)) << 4));
    const uint8_t index11 = uint8_t((((packed1 >> 8) >> shift) & uint16_t(0x0F)) | ((((high1 >> 8) >> high_shift) & uint16_t(1)) << 4));
    if (element % 32 == 0) {
        const uint8_t scale_low = (iqks_load_u8(block_offset + 2 + ib32_scale % 4) >> (4 * (ib32_scale / 4))) & uint8_t(0x0F);
        const uint8_t scale_high = uint8_t((iqks_load_u16(block_offset) >> (2 * ib32_scale)) & uint16_t(3));
        d_scale = float(int(scale_low | (scale_high << 4)) - 32);
    }

    value0 = pack32(i8vec4(kvalues_iq2_kl[2 * index00 + (pos32 & 1)],
                           kvalues_iq2_kl[2 * index00 + ((pos32 + 1) & 1)],
                           kvalues_iq2_kl[2 * index01 + ((pos32 + 2) & 1)],
                           kvalues_iq2_kl[2 * index01 + ((pos32 + 3) & 1)]));
    value1 = pack32(i8vec4(kvalues_iq2_kl[2 * index10 + (next_pos32 & 1)],
                           kvalues_iq2_kl[2 * index10 + ((next_pos32 + 1) & 1)],
                           kvalues_iq2_kl[2 * index11 + ((next_pos32 + 2) & 1)],
                           kvalues_iq2_kl[2 * index11 + ((next_pos32 + 3) & 1)]));
#else
    value0 = uint(iqks_value4(block_offset, element));
    value1 = uint(iqks_value4(block_offset, element + 16));
#endif
}

void block_a_to_shmem(const uint buf_ib, const uint row_offset, const uint ib, const uint iqs) {
    const uint ib32 = ib % 8;
    const uint block_offset = row_offset + IQK_ROW_META_SIZE + (ib / 8) * IQK_BLOCK_SIZE;
    const uint element = 32 * ib32 + 4 * iqs;
    uint value0;
    uint value1;
    float d_scale;
    mmq_iqks_pack8(block_offset, element, value0, value1, d_scale);
    buf_a[buf_ib].qs[iqs]     = int32_t(value0);
    buf_a[buf_ib].qs[iqs + 4] = int32_t(value1);

    if (iqs == 0) {
        buf_a[buf_ib].d_scales = FLOAT_TYPEV2(iqks_row_scale(row_offset) * d_scale);
    }
}
#endif

#if defined(DATA_A_IQ2_K) || defined(DATA_A_IQ3_K) || defined(DATA_A_IQ4_K) || defined(DATA_A_IQ5_K) || defined(DATA_A_IQ6_K) || defined(DATA_A_IQK_ROW)
void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;

    [[unroll]] for (uint iqs = 0; iqs < 8; ++iqs) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t sum0 = 0;
    int32_t sum1 = 0;
    [[unroll]] for (uint iqs = 0; iqs < 4; ++iqs) {
        sum0 += dotPacked4x8EXT(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
        sum1 += dotPacked4x8EXT(cache_a[ib_a].qs[iqs + 4], cache_b.qs[iqs + 4]);
    }

    return ACC_TYPE(float(cache_b.ds.x) * (float(cache_a[ib_a].d_scales.x) * float(sum0) + float(cache_a[ib_a].d_scales.y) * float(sum1)));
}
#endif

// For k-quants, ib and iqs still assume 32-wide blocks, but k-quants are 256-wide
// iqs still refers to a 32-bit integer, meaning 0..7 for 32-wide quants
#if defined(DATA_A_Q2_K)
// 4-byte loads for Q2_K blocks (84 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs * QUANT_R_MMQ;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;

    // Repack 4x4 quants into one int
    const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x03030303;
    const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x03030303;
    const uint32_t vals2 = (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x03030303;
    const uint32_t vals3 = (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x03030303;

    buf_a[buf_ib].qs[iqs] = vals0 | (vals1 << 2) | (vals2 << 4) | (vals3 << 6);

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPEV2(data_a_packed32[ib_k].dm);
        buf_a[buf_ib].scales = unpack8(uint32_t(data_a_packed16[ib_k].scales[iqs_k / 8])).xy; // vec4 used due to #12147
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;
    cache_a[reg_ib].scales = buf_a[buf_ib].scales;

    [[unroll]] for (uint iqs = 0; iqs < 2; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t sum_d = 0;
    int32_t sum_m = 0;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const uint8_t scale = cache_a[ib_a].scales[iqs / 4];
        const int32_t scale_m = int32_t(scale >> 4) * 0x01010101; // Duplicate 8-bit value across 32-bits.
        const int32_t qs_a = int32_t((cache_a[ib_a].qs[iqs / 4] >> ((iqs % 4) * 2)) & 0x03030303);

        sum_d += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]) * (scale & 0xF);
        sum_m += dotPacked4x8EXT(scale_m, cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_b.ds.x) * (float(cache_a[ib_a].dm.x) * float(sum_d) - float(cache_a[ib_a].dm.y) * float(sum_m)));
}
#endif

#if defined(DATA_A_Q3_K)
// 2-byte loads for Q3_K blocks (110 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint hm_idx = iqs * QUANT_R_MMQ;
    const uint iqs_k = (ib % 8) * 8 + hm_idx;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;
    const uint hm_shift = iqs_k / 8;

    // Repack 2x4 quants into one int
    // Add the 3rd bit instead of subtracting it to allow packing the quants
    // vec4 for unpack8 used due to #12147
    const i8vec2 vals00 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2        ] >> qs_shift) & uint16_t(0x0303)))).xy |
                          unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2    ] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
    const i8vec2 vals01 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 1    ] >> qs_shift) & uint16_t(0x0303)))).xy |
                          unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 1] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
    const i8vec2 vals10 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 2    ] >> qs_shift) & uint16_t(0x0303)))).xy |
                          unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 2] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
    const i8vec2 vals11 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 3    ] >> qs_shift) & uint16_t(0x0303)))).xy |
                          unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 3] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
    buf_a[buf_ib].qs[iqs] = pack32(u8vec4(vals00.x, vals00.y, vals01.x, vals01.y)) |
                           (pack32(u8vec4(vals10.x, vals10.y, vals11.x, vals11.y)) << 4);

    if (iqs == 0) {
        const uint is = iqs_k / 4;
        const i8vec2 scales = i8vec2(unpack8(uint32_t(((data_a_packed16[ib_k].scales[(is % 8      ) / 2] >> (4 * (is / 8))) & 0x0F0F) |
                                                     (((data_a_packed16[ib_k].scales[(8 + (is % 4)) / 2] >> (2 * (is / 4))) & 0x0303) << 4))).xy); // vec4 used due to #12147

        buf_a[buf_ib].d_scales = FLOAT_TYPEV2(float(data_a_packed16[ib_k].d) * vec2(scales - 32));
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    float result = 0.0;
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        // Subtract 4 from the quants to correct the 3rd bit offset
        const int32_t qs_a = pack32(unpack8(int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F)) - int8_t(4));

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[0]) * float(q_sum);
    q_sum = 0;

    [[unroll]] for (uint iqs = 4; iqs < 8; iqs++) {
        const int32_t qs_a = pack32(unpack8(int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F)) - int8_t(4));

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[1]) * float(q_sum);

    return ACC_TYPE(float(cache_b.ds.x) * result);
}
#endif

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
// 4-byte loads for Q4_K blocks (144 bytes) and Q5_K blocks (176 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs * QUANT_R_MMQ;

    const uint qs_idx = (iqs_k / 16) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 16) / 8) * 4;

    // Repack 2x4 quants into one int
#if defined(DATA_A_Q4_K)
    const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F;

    buf_a[buf_ib].qs[iqs] = vals0 | (vals1 << 4);
#else // defined(DATA_A_Q5_K)
    const uint qh_idx = iqs * QUANT_R_MMQ;
    const uint qh_shift = iqs_k / 8;

    buf_a[buf_ib].qs[iqs] = int32_t(((data_a_packed32[ib_k].qs[qs_idx] >> qs_shift) & 0x0F0F0F0F) |
                                   (((data_a_packed32[ib_k].qh[qh_idx] >> qh_shift) & 0x01010101) << 4));
#endif

    if (iqs == 0) {
        // Scale index
        const uint is = iqs_k / 8;
        u8vec2 scale_dm;
        if (is < 4) {
            scale_dm = u8vec2(data_a[ib_k].scales[is] & 0x3F, data_a[ib_k].scales[is + 4] & 0x3F);
        } else {
            scale_dm = u8vec2((data_a[ib_k].scales[is+4] & 0xF) | ((data_a[ib_k].scales[is-4] & 0xC0) >> 2),
                              (data_a[ib_k].scales[is+4] >>  4) | ((data_a[ib_k].scales[is  ] & 0xC0) >> 2));
        }

        buf_a[buf_ib].dm = FLOAT_TYPEV2(vec2(data_a_packed32[ib_k].dm) * vec2(scale_dm));
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 8 / QUANT_R_MMQ; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
#if defined(DATA_A_Q4_K)
        const int32_t qs_a = int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F);
#else // defined(DATA_A_Q5_K)
        const int32_t qs_a = cache_a[ib_a].qs[iqs];
#endif

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_b.ds.x) * float(cache_a[ib_a].dm.x) * float(q_sum) - float(cache_a[ib_a].dm.y) * float(cache_b.ds.y));
}
#endif

#if defined(DATA_A_Q6_K)
// 2-byte loads for Q6_K blocks (210 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint ql_idx = (iqs_k / 32) * 16 + iqs_k % 16;
    const uint ql_shift = ((iqs_k % 32) / 16) * 4;

    const uint qh_idx = (iqs_k / 32) * 8 + iqs;
    const uint qh_shift = ((iqs_k % 32) / 8) * 2;

    const i8vec2 vals00 = (unpack8(int32_t((data_a_packed16[ib_k].ql[ql_idx * 2    ] >> ql_shift) & uint16_t(0x0F0F))).xy |
                          unpack8(int32_t(((data_a_packed16[ib_k].qh[qh_idx * 2    ] >> qh_shift) & uint16_t(0x0303)) << 4)).xy) - int8_t(32);
    const i8vec2 vals01 = (unpack8(int32_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 1] >> ql_shift) & uint16_t(0x0F0F))).xy |
                          unpack8(int32_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 1] >> qh_shift) & uint16_t(0x0303)) << 4)).xy) - int8_t(32);
    buf_a[buf_ib].qs[iqs] = pack32(i8vec4(vals00.x, vals00.y, vals01.x, vals01.y));

    if (iqs == 0) {
        const uint is = iqs_k / 4;
        const i8vec2 scales = unpack8(int32_t(data_a_packed16[ib_k].scales[is / 2])).xy;

        buf_a[buf_ib].d_scales = FLOAT_TYPEV2(float(data_a_packed16[ib_k].d) * vec2(scales));
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    float result = 0.0;
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[0]) * float(q_sum);
    q_sum = 0;

    [[unroll]] for (uint iqs = 4; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[1]) * float(q_sum);

    return ACC_TYPE(float(cache_b.ds.x) * result);
}
#endif

#if defined(DATA_A_IQ3_S)
// 2-byte loads for IQ3_S blocks (110 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;

    // grid indices for qs[2 * iqs] and qs[2 * iqs + 1]
    const uint qs    = uint(data_a_packed16[ib_k].qs[ib32 * 4 + iqs]);
    // their two high index bits
    const uint qh    = uint(data_a_packed16[ib_k].qh[ib32 / 2]) >> ((ib32 & 1) * 8 + 2 * iqs);
    // one sign bit per value, 8 values
    const uint signs = uint(data_a_packed16[ib_k].signs[ib32 * 2 + iqs / 2]) >> ((iqs & 1) * 8);

    // grid holds 4 values of 1..15, one per byte
    const ivec4 vals0 = ivec4(unpack8(iq3s_grid[( qs       & 0xFF) | ((qh & 1) << 8)]));
    const ivec4 vals1 = ivec4(unpack8(iq3s_grid[((qs >> 8) & 0xFF) | ((qh & 2) << 7)]));

    // negate with (v ^ -s) - -s to avoid branches
    const ivec4 m0 = -(ivec4(signs,      signs >> 1, signs >> 2, signs >> 3) & 1);
    const ivec4 m1 = -(ivec4(signs >> 4, signs >> 5, signs >> 6, signs >> 7) & 1);

    buf_a[buf_ib].qs[2 * iqs    ] = pack32(i8vec4((vals0 ^ m0) - m0));
    buf_a[buf_ib].qs[2 * iqs + 1] = pack32(i8vec4((vals1 ^ m1) - m1));

    if (iqs == 0) {
        const uint scale = (uint(data_a_packed16[ib_k].scales[ib32 / 4]) >> ((ib32 & 3) * 4)) & 0xF;

        buf_a[buf_ib].d = FLOAT_TYPE(float(data_a_packed16[ib_k].d) * float(1 + 2 * scale));
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d = buf_a[buf_ib].d;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        q_sum += dotPacked4x8EXT(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_a[ib_a].d) * float(cache_b.ds.x) * float(q_sum));
}
#endif

void block_b_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint ib_outer = ib / 4;
        const uint ib_inner = ib % 4;

        if (iqs == 0) {
            buf_b[buf_ib].ds = FLOAT_TYPEV2(data_b[ib_outer].ds[ib_inner]);
        }

        const ivec4 values = data_b[ib_outer].qs[ib_inner * 2 + iqs];
        buf_b[buf_ib].qs[iqs * 4    ] = values.x;
        buf_b[buf_ib].qs[iqs * 4 + 1] = values.y;
        buf_b[buf_ib].qs[iqs * 4 + 2] = values.z;
        buf_b[buf_ib].qs[iqs * 4 + 3] = values.w;
    } else {
        if (iqs == 0) {
            buf_b[buf_ib].ds = FLOAT_TYPEV2(0.0f);
        }

        buf_b[buf_ib].qs[iqs * 4    ] = 0;
        buf_b[buf_ib].qs[iqs * 4 + 1] = 0;
        buf_b[buf_ib].qs[iqs * 4 + 2] = 0;
        buf_b[buf_ib].qs[iqs * 4 + 3] = 0;
    }
}

void block_b_to_registers(const uint ib) {
    cache_b.ds = buf_b[ib].ds;
    [[unroll]] for (uint iqs = 0; iqs < BK / 4; iqs++) {
        cache_b.qs[iqs] = buf_b[ib].qs[iqs];
    }
}
