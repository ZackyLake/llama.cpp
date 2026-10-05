
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require

uint32_t u4x4_to_u8x4(uint32_t x) {
    x = (x | (x << 8)) & 0x00ff00ffu;
    return (x | (x << 4)) & 0x0f0f0f0fu;
}

uint32_t u2x4_to_u8x4(uint16_t x) {
    const uint32_t t = (x | (x << 6)) & uint16_t(0x3333);
    return (t | (t << 12u)) & 0x03030303u;
}

uint32_t u2x4_to_u8x4_sub32(uint16_t x) {
    const uint16_t t = (x | (x << 6)) & uint16_t(0x3333);
    return (t * 0x10010u + 0xe0e0e0e0u) & 0xf0f0f0f0u;
}

uint32_t test_bit_to_mask(uint32_t dat, int idx) {
    return uint32_t(bitfieldExtract(int32_t(dat), idx, 1));
}

uint32_t test_bit_to_get_bit(uint32_t dat, int idx_test, int idx_get) {
    const uint32_t dat_shift = idx_test == idx_get ? dat : 
        (idx_test < idx_get ? dat << (idx_get - idx_test) : dat >> (idx_test - idx_get));
    return dat_shift & (1u << idx_get);
}
uint16_t test_bit_to_get_bit(uint16_t dat, int idx_test, int idx_get) {
    const uint16_t dat_shift = idx_test == idx_get ? dat : 
        (idx_test < idx_get ? dat << (idx_get - idx_test) : dat >> (idx_test - idx_get));
    return dat_shift & (uint16_t(1) << idx_get);
}
uint8_t test_bit_to_get_bit(uint8_t dat, int idx_test, int idx_get) {
    const uint8_t dat_shift = idx_test == idx_get ? dat : 
        (idx_test < idx_get ? dat << (idx_get - idx_test) : dat >> (idx_test - idx_get));
    // intermediates are free to be promoted to i16
    return dat_shift & (uint8_t(1) << idx_get);
}

uint8_t test_bit_to_get_high4(uint32_t dat, int idx, uint8_t val) {
    const uint32_t shift = test_bit_to_get_bit(dat, idx, 2);
    return uint8_t(bitfieldExtract(uint32_t(val), int32_t(shift), 4));
    // const uint32_t dat_shift = idx == 2 ? dat : (idx < 2 ? dat << (2 - idx) : dat >> (idx - 2));
    // return uint8_t(bitfieldExtract(uint32_t(val), int32_t(dat_shift & 4u), 4));
    // return (val >> (dat_shift & 4u)) & uint8_t(0x0F);
}

uint16_t shift_to_get_high4(uint16_t shift, uint16_t val) {
    return (val >> shift) & uint16_t(0x0F);
    // return uint16_t(bitfieldExtract(uint32_t(val), shift, 4));
}
uint8_t shift_to_get_high4(uint16_t shift, uint8_t val) {
    return (val >> shift) & uint8_t(0x0F);
    // return uint8_t(bitfieldExtract(uint32_t(val), shift, 4));
}

uint16_t test_bit_to_get_high4(uint16_t dat, int idx, uint16_t val) {
    const uint16_t shift = test_bit_to_get_bit(dat, idx, 2);
    return shift_to_get_high4(shift, val);
}
uint8_t test_bit_to_get_high4(uint16_t dat, int idx, uint8_t val) {
    const uint16_t shift = test_bit_to_get_bit(dat, idx, 2);
    return shift_to_get_high4(shift, val);

    // arithmetic/logic execution dtype is at least 16bit and src1 does not accept byte
    // LLVM's shift uses same dtype for all srcs, so promote val to u16
}

uint16_t test_bit_to_get_high4(uint8_t dat, int idx, uint16_t val) {
    const uint16_t shift = test_bit_to_get_bit(dat, idx, 2);
    return shift_to_get_high4(shift, val);
}
uint8_t test_bit_to_get_high4(uint8_t dat, int idx, uint8_t val) {
    const uint16_t shift = test_bit_to_get_bit(dat, idx, 2);
    return shift_to_get_high4(shift, val);
}

#if defined(DATA_A_IQ2_K) || defined(DATA_A_IQ2_KS)
int32_t unpack_iq2_k(uint32_t values, bool is_hi_table) {
    const uint table_offset = is_hi_table ? 256u : 0u;
    const uint index = table_offset + dotPacked4x8EXT(values, 0x40100401u);
    return int32_t(iq2k_table[index]);
}
#endif

#if defined(DATA_A_IQ3_K) || defined(DATA_A_IQ3_KS)
#if defined(IQK_MMQ_LUT_LARGE)
int32_t unpack_iq3_k(uint32_t ql, uint32_t qh, bool is_hi_table) {
    const uint indexes = (ql & 0x03030303u) | ((qh & 0x01010101u) << 2u);
    const uint index = (dotPacked4x8EXT(indexes, 0x40080100u) << 3u) | (indexes & 0x7u);
    return int32_t(iq3k_table4[index] + (is_hi_table ? 0x04040404u : 0u));
}

int32_t unpack_iq3_k(uint32_t ql, uint32_t qh, uint16_t shift_h, bool is_hi_table) {
    const uint16_t shift_l = (shift_h & uint16_t(3)) << 1;
    return unpack_iq3_k(ql >> shift_l, qh >> shift_h, is_hi_table);
}

int32_t unpack_iq3_k(uint32_t ql, uint32_t qh, uint16_t shift_h, uint16_t extra_bit01, int hi_table_bit_idx) {
    const uint16_t shift_l = (shift_h & uint16_t(3)) << 1;
    const bool is_hi_table = bool(test_bit_to_get_bit(extra_bit01, hi_table_bit_idx, 6));
    return unpack_iq3_k(ql >> shift_l, qh >> shift_h, is_hi_table);
}

#else
int32_t unpack_iq3_k(uint32_t ql, uint32_t qh, bool is_hi_table) {
    const uint indexes = (ql & 0x03030303u) | ((qh & 0x01010101u) << 2u);
    const uint table_offset = is_hi_table ? 64u : 0u;
    const uint index0 = table_offset + dotPacked4x8EXT(indexes & 0xffffu, 0x00000801u); // WA for intel bug
    const uint index1 = table_offset + dotPacked4x8EXT(indexes, 0x08010000u);
    return int32_t(pack32(u16vec2(iq3k_table[index0], iq3k_table[index1])));
}
int32_t unpack_iq3_k(uint32_t ql, uint32_t qh, uint16_t shift_h, bool is_hi_table) {
    const uint16_t shift_l = (shift_h & uint16_t(3)) << 1;
    const uint indexes = ((ql >> shift_l) & 0x03030303u) |
                         (((qh >> shift_h) & 0x01010101u) << 2u);
    const uint table_offset = is_hi_table ? 64u : 0u;
    const uint index0 = table_offset + dotPacked4x8EXT(indexes & 0xffffu, 0x00000801u); // WA for intel bug
    const uint index1 = table_offset + dotPacked4x8EXT(indexes, 0x08010000u);
    return int32_t(pack32(u16vec2(iq3k_table[index0], iq3k_table[index1])));
}
int32_t unpack_iq3_k(uint32_t ql, uint32_t qh, uint16_t shift_h, uint16_t extra_bit01, int hi_table_bit_idx) {
    const uint16_t shift_l = (shift_h & uint16_t(3)) << 1;
    const uint indexes = ((ql >> shift_l) & 0x03030303u) |
                         (((qh >> shift_h) & 0x01010101u) << 2u);
    const uint table_offset = test_bit_to_get_bit(extra_bit01, hi_table_bit_idx, 6);
    const uint index0 = table_offset + dotPacked4x8EXT(indexes & 0xffffu, 0x00000801u); // WA for intel bug
    const uint index1 = table_offset + dotPacked4x8EXT(indexes, 0x08010000u);
    return int32_t(pack32(u16vec2(iq3k_table[index0], iq3k_table[index1])));
}
#endif
#endif

#if defined(DATA_A_IQ4_K) || defined(DATA_A_IQ4_KSS) || defined(DATA_A_IQ4_KS)
int32_t unpack_iq4_k(uint32_t values, bool is_hi_table) {
    const uint table_base = is_hi_table ? 256u : 0u;
    const uint index0 = table_base + dotPacked4x8EXT(values & 0x00000F0Fu, 0x00001001u);
    const uint index1 = table_base + dotPacked4x8EXT(values & 0x0F0F0000u, 0x10010000u);
    return int32_t(pack32(u16vec2(iq4k_table[index0], iq4k_table[index1])));
}

int32_t unpack_iq4_k(uint32_t values, uint shift, bool is_hi_table) {
    return unpack_iq4_k(values >> shift, is_hi_table);
}

#endif

#if defined(DATA_A_IQ5_K) || defined(DATA_A_IQ5_KS)
int32_t unpack_iq5_k(uint32_t values, uint32_t qh, uint shift, uint table_offset) {
    const uint32_t low = (values >> (4 * (shift & 1))) & 0x0F0F0F0Fu;
    const uint32_t high = ((qh >> shift) & 0x01010101u) << 4u;
    const u8vec4 indexes = unpack8(low | high);
    return pack32(i8vec4(kvalues_iq5_k[indexes.x + table_offset],
                         kvalues_iq5_k[indexes.y + table_offset],
                         kvalues_iq5_k[indexes.z + table_offset],
                         kvalues_iq5_k[indexes.w + table_offset]));
}
#endif

#if defined(DATA_A_IQ2_KL)
i32vec2 unpack_iq2_kl(uint values, uint high, uint16_t ib32) {
    const uint16_t shift_values = (ib32 & uint16_t(1)) << 2u;
    const u8vec4 indexes = (unpack8((values >> shift_values) & 0x0F0F0F0Fu) << 1u) |
                           (unpack8((high >> ib32) & 0x01010101u) << 5u);
    return i32vec2(int32_t(pack32(u16vec2(iq2kl_table[indexes.x], iq2kl_table[indexes.y]))),
                   int32_t(pack32(u16vec2(iq2kl_table[indexes.z], iq2kl_table[indexes.w]))));
}
#endif

#if defined(DATA_A_IQ2_K)
void prepare_a_k32_iter(uint ib_k, uint8_t ib32, out int32_t qs_a[8], out vec2 a_scales) {
    const uint8_t qs_idx = bool(ib32 & uint8_t(4)) ? uint8_t(8) : uint8_t(0);
    const uint8_t ib32_shift = ib32 << 1;
    const uint8_t shift = ib32_shift & uint8_t(6);
    const bool is_hi_table0 = bool((data_a[ib_k].extra >> ib32_shift) & uint16_t(1));
    const bool is_hi_table1 = bool((data_a[ib_k].extra >> (ib32_shift + uint8_t(1))) & uint16_t(1));

    qs_a[0] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx    ] >> shift) & 0x03030303, is_hi_table0);
    qs_a[1] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 1] >> shift) & 0x03030303, is_hi_table0);
    qs_a[2] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 2] >> shift) & 0x03030303, is_hi_table0);
    qs_a[3] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 3] >> shift) & 0x03030303, is_hi_table0);
    qs_a[4] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 4] >> shift) & 0x03030303, is_hi_table1);
    qs_a[5] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 5] >> shift) & 0x03030303, is_hi_table1);
    qs_a[6] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 6] >> shift) & 0x03030303, is_hi_table1);
    qs_a[7] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 7] >> shift) & 0x03030303, is_hi_table1);
    const uint8_t scale = data_a[ib_k].scales[ib32];
    const float d = float(data_a[ib_k].d);
    a_scales = vec2(d * float(int32_t(scale & uint8_t(0x0F)) - 8),
                    d * float(int32_t(scale >> uint8_t(4)) - 8));
}

void prepare_a_k16x_iter(uint ib_k, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t laneid = uint16_t(gl_SubgroupInvocationID);
    const uint16_t subid = laneid >> 2;
    const bool is_high_16 = bool(subid & uint16_t(1));
    const bool is_high_half = bool(subid & uint16_t(2));
    const uint16_t qs_idx = laneid & uint16_t(0x0c);
    const uvec4 qs = uvec4(data_a_packed32[ib_k].qs[qs_idx     ],
                           data_a_packed32[ib_k].qs[qs_idx + 1u],
                           data_a_packed32[ib_k].qs[qs_idx + 2u],
                           data_a_packed32[ib_k].qs[qs_idx + 3u]);
    const uint32_t scale_packed = data_a_packed32[ib_k].scales[is_high_half ? 1u : 0u];
    const uint32_t scale_shifted = scale_packed >> (is_high_16 ? 4 : 0);
    const uint32_t scale_sub8 = ((scale_shifted | 0xF0F0F0F0u) - 0x78787878u) ^ 0x80808080u;
    a_scales = unpack8(int32_t(scale_sub8)); // unpack8(int32_t(scale_shifted & 0x0F0F0F0Fu)) - i8vec4(-8)

    [[unroll]] for (uint16_t i = uint16_t(0); i < uint16_t(4); ++i) {
        const uint16_t shift = i << 1u;
        const uint16_t extra_shift_idx = i * uint16_t(2) + (is_high_half ? uint16_t(8) : uint16_t(0)) + uint16_t(is_high_16);
        const bool is_hi_table = bool((data_a[ib_k].extra >> extra_shift_idx) & uint16_t(1));

        qs_a[i] = i32vec4(unpack_iq2_k((qs.x >> shift) & 0x03030303u, is_hi_table),
                          unpack_iq2_k((qs.y >> shift) & 0x03030303u, is_hi_table),
                          unpack_iq2_k((qs.z >> shift) & 0x03030303u, is_hi_table),
                          unpack_iq2_k((qs.w >> shift) & 0x03030303u, is_hi_table));
    }
}

void prepare_a_k16x4_iter_shifted(uint ib_k, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t laneid = uint16_t(gl_SubgroupInvocationID);
    const uint16_t subid = laneid >> uint16_t(2);
    const bool is_high_16 = bool(subid & uint16_t(1));
    const bool is_high_half = bool(subid & uint16_t(2));
    const uint16_t qs_idx = laneid & uint16_t(0x0c);
    const uvec4 qs = uvec4(data_a_packed32[ib_k].qs[qs_idx     ],
                           data_a_packed32[ib_k].qs[qs_idx + 1u],
                           data_a_packed32[ib_k].qs[qs_idx + 2u],
                           data_a_packed32[ib_k].qs[qs_idx + 3u]);
    const uint32_t scale_packed = data_a_packed32[ib_k].scales[is_high_half ? 1u : 0u];
    const uint32_t scale_shifted = scale_packed >> (is_high_16 ? 4 : 0);
    const uint32_t scale_sub8 = ((scale_shifted | 0xF0F0F0F0u) - 0x78787878u) ^ 0x80808080u;
    a_scales = unpack8(int32_t(scale_sub8)); // unpack8(int32_t(scale_shifted & 0x0F0F0F0Fu)) - i8vec4(-8)
    const uint16_t row = laneid & uint16_t(3);

    [[unroll]] for (uint16_t i = uint16_t(0); i < uint16_t(4); ++i) {
        const uint16_t preshift_i = i ^ row;
        const uint16_t shift = preshift_i << 1u;
        const uint16_t extra_shift_idx = preshift_i * uint16_t(2) + (is_high_half ? uint16_t(8) : uint16_t(0)) + uint16_t(is_high_16);
        const bool is_hi_table = bool((data_a[ib_k].extra >> extra_shift_idx) & uint16_t(1));

        qs_a[i] = i32vec4(unpack_iq2_k((qs.x >> shift) & 0x03030303u, is_hi_table),
                          unpack_iq2_k((qs.y >> shift) & 0x03030303u, is_hi_table),
                          unpack_iq2_k((qs.z >> shift) & 0x03030303u, is_hi_table),
                          unpack_iq2_k((qs.w >> shift) & 0x03030303u, is_hi_table));
    }
}
#define SG_A_SHAPE_16x4
#endif

#if defined(DATA_A_IQ3_K)
void prepare_a_k32_iter(uint ib_k, uint8_t ib32, out int32_t qs_a[8], out vec2 a_scales) {
    const uint q_idx = bool(ib32 & uint8_t(4)) ? 16u : 0u;
    const uint8_t ib32_shift = ib32 << 1;
    const uint16_t extra_bit01 = data_a[ib_k].extra >> ib32_shift;

    const uint ql0 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx +  0], data_a_packed16[ib_k].qs[q_idx +  1]));
    const uint ql1 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx +  2], data_a_packed16[ib_k].qs[q_idx +  3]));
    const uint ql2 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx +  4], data_a_packed16[ib_k].qs[q_idx +  5]));
    const uint ql3 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx +  6], data_a_packed16[ib_k].qs[q_idx +  7]));
    const uint ql4 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx +  8], data_a_packed16[ib_k].qs[q_idx +  9]));
    const uint ql5 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx + 10], data_a_packed16[ib_k].qs[q_idx + 11]));
    const uint ql6 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx + 12], data_a_packed16[ib_k].qs[q_idx + 13]));
    const uint ql7 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx + 14], data_a_packed16[ib_k].qs[q_idx + 15]));
    const uint qh0 = pack32(u16vec2(data_a_packed16[ib_k].qh[ 0], data_a_packed16[ib_k].qh[ 1]));
    const uint qh1 = pack32(u16vec2(data_a_packed16[ib_k].qh[ 2], data_a_packed16[ib_k].qh[ 3]));
    const uint qh2 = pack32(u16vec2(data_a_packed16[ib_k].qh[ 4], data_a_packed16[ib_k].qh[ 5]));
    const uint qh3 = pack32(u16vec2(data_a_packed16[ib_k].qh[ 6], data_a_packed16[ib_k].qh[ 7]));
    const uint qh4 = pack32(u16vec2(data_a_packed16[ib_k].qh[ 8], data_a_packed16[ib_k].qh[ 9]));
    const uint qh5 = pack32(u16vec2(data_a_packed16[ib_k].qh[10], data_a_packed16[ib_k].qh[11]));
    const uint qh6 = pack32(u16vec2(data_a_packed16[ib_k].qh[12], data_a_packed16[ib_k].qh[13]));
    const uint qh7 = pack32(u16vec2(data_a_packed16[ib_k].qh[14], data_a_packed16[ib_k].qh[15]));

    qs_a[0] = unpack_iq3_k(ql0, qh0, uint16_t(ib32), extra_bit01, 0);
    qs_a[1] = unpack_iq3_k(ql1, qh1, uint16_t(ib32), extra_bit01, 0);
    qs_a[2] = unpack_iq3_k(ql2, qh2, uint16_t(ib32), extra_bit01, 0);
    qs_a[3] = unpack_iq3_k(ql3, qh3, uint16_t(ib32), extra_bit01, 0);
    qs_a[4] = unpack_iq3_k(ql4, qh4, uint16_t(ib32), extra_bit01, 1);
    qs_a[5] = unpack_iq3_k(ql5, qh5, uint16_t(ib32), extra_bit01, 1);
    qs_a[6] = unpack_iq3_k(ql6, qh6, uint16_t(ib32), extra_bit01, 1);
    qs_a[7] = unpack_iq3_k(ql7, qh7, uint16_t(ib32), extra_bit01, 1);

    const uint16_t scale_l = uint16_t(data_a[ib_k].scales_l[ib32]);
    const uint16_t sign01 = data_a[ib_k].scales_h >> ib32_shift;
    const bool sign0 = bool(sign01 & uint16_t(1));
    const bool sign1 = bool(sign01 & uint16_t(2));
    const int16_t scale0 = int16_t((scale_l & uint16_t(0x0F)) * uint16_t(2) + uint16_t(1));
    const int16_t scale1 = int16_t((scale_l >> uint16_t(3)) | uint16_t(1));
    const float d = data_a[ib_k].d;
    a_scales = vec2(d * float(sign0 ? -scale0 : scale0),
                    d * float(sign1 ? -scale1 : scale1));
}

void prepare_a_k16x_iter(uint ib_k, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t laneid = uint16_t(gl_SubgroupInvocationID);
    const uint16_t subid = laneid >> uint16_t(2);
    const bool is_high_16 = bool(subid & uint16_t(1));
    const bool is_high_half = bool(subid & uint16_t(2));
    const uint16_t qs_idx = (is_high_half ? uint16_t(16) : uint16_t(0)) +
                            (is_high_16 ? uint16_t(8) : uint16_t(0));
    const uint16_t qh_idx = is_high_16 ? uint16_t(8) : uint16_t(0);
    const uint ql0 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_idx     ], data_a_packed16[ib_k].qs[qs_idx + 1u]));
    const uint ql1 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_idx + 2u], data_a_packed16[ib_k].qs[qs_idx + 3u]));
    const uint ql2 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_idx + 4u], data_a_packed16[ib_k].qs[qs_idx + 5u]));
    const uint ql3 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_idx + 6u], data_a_packed16[ib_k].qs[qs_idx + 7u]));
    const uint qh0 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_idx     ], data_a_packed16[ib_k].qh[qh_idx + 1u]));
    const uint qh1 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_idx + 2u], data_a_packed16[ib_k].qh[qh_idx + 3u]));
    const uint qh2 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_idx + 4u], data_a_packed16[ib_k].qh[qh_idx + 5u]));
    const uint qh3 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_idx + 6u], data_a_packed16[ib_k].qh[qh_idx + 7u]));
    const uint16_t scale_shift = (is_high_half ? uint16_t(8) : uint16_t(0)) + uint16_t(is_high_16);
    const uint scale_idx = is_high_half ? 2u : 0u;
    const uint scale_packed = pack32(u16vec2(data_a_packed16[ib_k].scales_l[scale_idx],
                                             data_a_packed16[ib_k].scales_l[scale_idx + 1u]));
    const uint scale_nibbles = (scale_packed >> (is_high_16 ? 4u : 0u)) & 0x0F0F0F0Fu;
    const uint scale_shifted = (scale_nibbles << 1u) | 0x01010101u;
    const uint16_t scale_h = data_a_packed16[ib_k].scales_h >> scale_shift;
    const uint scale_sign = ((scale_h & uint16_t(0x55)) * 0x41041u) & 0x01010101u;
    a_scales = unpack8(int32_t((scale_sign * 254u) ^ scale_shifted));

    [[unroll]] for (uint16_t i = uint16_t(0); i < uint16_t(4); ++i) {
        const uint16_t shift_h = i + (is_high_half ? uint16_t(4) : uint16_t(0));
        const uint16_t shift_l = i << 1u;
        const uint16_t extra_shift_idx = i * uint16_t(2) + scale_shift;
        const bool is_hi_table = bool((data_a[ib_k].extra >> extra_shift_idx) & uint16_t(1));

        qs_a[i] = i32vec4(unpack_iq3_k(ql0 >> shift_l, qh0 >> shift_h, is_hi_table),
                          unpack_iq3_k(ql1 >> shift_l, qh1 >> shift_h, is_hi_table),
                          unpack_iq3_k(ql2 >> shift_l, qh2 >> shift_h, is_hi_table),
                          unpack_iq3_k(ql3 >> shift_l, qh3 >> shift_h, is_hi_table));
    }
}

void prepare_a_k16x4_iter_shifted(uint ib_k, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t laneid = uint16_t(gl_SubgroupInvocationID);
    const uint16_t subid = laneid >> uint16_t(2);
    const bool is_high_16 = bool(subid & uint16_t(1));
    const bool is_high_half = bool(subid & uint16_t(2));
    const uint16_t qs_idx = (is_high_half ? uint16_t(16) : uint16_t(0)) +
                            (is_high_16 ? uint16_t(8) : uint16_t(0));
    const uint16_t qh_idx = is_high_16 ? uint16_t(8) : uint16_t(0);
    const uint ql0 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_idx     ], data_a_packed16[ib_k].qs[qs_idx + 1u]));
    const uint ql1 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_idx + 2u], data_a_packed16[ib_k].qs[qs_idx + 3u]));
    const uint ql2 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_idx + 4u], data_a_packed16[ib_k].qs[qs_idx + 5u]));
    const uint ql3 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_idx + 6u], data_a_packed16[ib_k].qs[qs_idx + 7u]));
    const uint qh0 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_idx     ], data_a_packed16[ib_k].qh[qh_idx + 1u]));
    const uint qh1 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_idx + 2u], data_a_packed16[ib_k].qh[qh_idx + 3u]));
    const uint qh2 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_idx + 4u], data_a_packed16[ib_k].qh[qh_idx + 5u]));
    const uint qh3 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_idx + 6u], data_a_packed16[ib_k].qh[qh_idx + 7u]));
    const uint16_t scale_shift = (is_high_half ? uint16_t(8) : uint16_t(0)) + uint16_t(is_high_16);
    const uint scale_idx = is_high_half ? 2u : 0u;
    const uint scale_packed = pack32(u16vec2(data_a_packed16[ib_k].scales_l[scale_idx],
                                             data_a_packed16[ib_k].scales_l[scale_idx + 1u]));
    const uint scale_nibbles = (scale_packed >> (is_high_16 ? 4u : 0u)) & 0x0F0F0F0Fu;
    const uint scale_shifted = (scale_nibbles << 1u) | 0x01010101u;
    const uint16_t scale_h = data_a_packed16[ib_k].scales_h >> scale_shift;
    const uint scale_sign = ((scale_h & uint16_t(0x55)) * 0x41041u) & 0x01010101u;
    a_scales = unpack8(int32_t((scale_sign * 254u) ^ scale_shifted));
    const uint16_t row = laneid & uint16_t(3);

    [[unroll]] for (uint16_t i = uint16_t(0); i < uint16_t(4); ++i) {
        const uint16_t preshift_i = i ^ row;
        const uint16_t shift_h = preshift_i + (is_high_half ? uint16_t(4) : uint16_t(0));
        const uint16_t shift_l = preshift_i << 1u;
        const uint16_t extra_shift_idx = preshift_i * uint16_t(2) + scale_shift;
        const bool is_hi_table = bool((data_a[ib_k].extra >> extra_shift_idx) & uint16_t(1));

        qs_a[i] = i32vec4(unpack_iq3_k(ql0 >> shift_l, qh0 >> shift_h, is_hi_table),
                          unpack_iq3_k(ql1 >> shift_l, qh1 >> shift_h, is_hi_table),
                          unpack_iq3_k(ql2 >> shift_l, qh2 >> shift_h, is_hi_table),
                          unpack_iq3_k(ql3 >> shift_l, qh3 >> shift_h, is_hi_table));

    }
}
#define SG_A_SHAPE_16x4
#endif

#if defined(DATA_A_IQ4_K)

void prepare_a_k32_iter(uint ib_k, uint8_t ib32, out int32_t qs_a[8], out vec2 a_scales) {
    const uint ib32_u = uint(ib32);
    const uint qs_idx = 4 * ib32_u;
    const bool is_hi_table0 = bool((data_a[ib_k].extra >> (2 * ib32_u)) & uint16_t(1));
    const bool is_hi_table1 = bool((data_a[ib_k].extra >> (2 * ib32_u + 1)) & uint16_t(1));

    qs_a[0] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx    ], 0, is_hi_table0);
    qs_a[1] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 1], 0, is_hi_table0);
    qs_a[2] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 2], 0, is_hi_table0);
    qs_a[3] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 3], 0, is_hi_table0);
    qs_a[4] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx    ], 4, is_hi_table1);
    qs_a[5] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 1], 4, is_hi_table1);
    qs_a[6] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 2], 4, is_hi_table1);
    qs_a[7] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 3], 4, is_hi_table1);
    const uint scales_h = uint(data_a[ib_k].scales_h[ib32_u / 2]) >> (4 * (ib32_u % 2));
    const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
    const int32_t scale0 = int32_t((scales_l & 0x0F) | ((scales_h << 4) & 0x30)) - 32;
    const int32_t scale1 = int32_t((scales_l >> 4) | ((scales_h << 2) & 0x30)) - 32;
    const float d = float(data_a[ib_k].d);
    a_scales = vec2(d * float(scale0), d * float(scale1));
}

void prepare_a_k32x2_iter_shifted(uint ib_k, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t row = uint16_t(gl_SubgroupInvocationID) & uint16_t(3);
    const uint16_t subid = uint16_t(gl_SubgroupInvocationID) >> 2u;
    const uint16_t ib32 = subid << 1u;
    const uint16_t qs_idx = ib32 << 2u;
    const uint16_t table_offset_bits = data_a[ib_k].extra >> (ib32 << 1u);

    const bool swap_qs_halves = bool(row & uint16_t(2));
    const uint16_t qs_idx0 = qs_idx + (swap_qs_halves ? uint16_t(4) : uint16_t(0));
    const uint16_t qs_idx1 = qs_idx + (swap_qs_halves ? uint16_t(0) : uint16_t(4));
    const uvec4 qs0 = uvec4(data_a_packed32[ib_k].qs[qs_idx0 + 0u],
                            data_a_packed32[ib_k].qs[qs_idx0 + 1u],
                            data_a_packed32[ib_k].qs[qs_idx0 + 2u],
                            data_a_packed32[ib_k].qs[qs_idx0 + 3u]);
    const uvec4 qs1 = uvec4(data_a_packed32[ib_k].qs[qs_idx1 + 0u],
                            data_a_packed32[ib_k].qs[qs_idx1 + 1u],
                            data_a_packed32[ib_k].qs[qs_idx1 + 2u],
                            data_a_packed32[ib_k].qs[qs_idx1 + 3u]);
    const uint16_t shift0 = (row & uint16_t(1)) << 2u;
    const uint16_t shift1 = ((row ^ uint16_t(1)) & uint16_t(1)) << 2u;
    const bool is_hi_table0 = bool((table_offset_bits >> row) & uint16_t(1));
    const bool is_hi_table1 = bool((table_offset_bits >> (row ^ uint16_t(1))) & uint16_t(1));
    const bool is_hi_table2 = bool((table_offset_bits >> (row ^ uint16_t(2))) & uint16_t(1));
    const bool is_hi_table3 = bool((table_offset_bits >> (row ^ uint16_t(3))) & uint16_t(1));

    qs_a[0] = i32vec4(unpack_iq4_k(qs0.x >> shift0, is_hi_table0),
                      unpack_iq4_k(qs0.y >> shift0, is_hi_table0),
                      unpack_iq4_k(qs0.z >> shift0, is_hi_table0),
                      unpack_iq4_k(qs0.w >> shift0, is_hi_table0));
    qs_a[1] = i32vec4(unpack_iq4_k(qs0.x >> shift1, is_hi_table1),
                      unpack_iq4_k(qs0.y >> shift1, is_hi_table1),
                      unpack_iq4_k(qs0.z >> shift1, is_hi_table1),
                      unpack_iq4_k(qs0.w >> shift1, is_hi_table1));
    qs_a[2] = i32vec4(unpack_iq4_k(qs1.x >> shift0, is_hi_table2),
                      unpack_iq4_k(qs1.y >> shift0, is_hi_table2),
                      unpack_iq4_k(qs1.z >> shift0, is_hi_table2),
                      unpack_iq4_k(qs1.w >> shift0, is_hi_table2));
    qs_a[3] = i32vec4(unpack_iq4_k(qs1.x >> shift1, is_hi_table3),
                      unpack_iq4_k(qs1.y >> shift1, is_hi_table3),
                      unpack_iq4_k(qs1.z >> shift1, is_hi_table3),
                      unpack_iq4_k(qs1.w >> shift1, is_hi_table3));

    const uint16_t scale_l = data_a_packed16[ib_k].scales_l[subid];
    const uint scale_lo = u4x4_to_u8x4(scale_l);
    const uint16_t scale_h = data_a[ib_k].scales_h[subid];
    const uint scale_hi = u2x4_to_u8x4_sub32(scale_h);
    a_scales = unpack8(int32_t(scale_lo | scale_hi));
}
#define SG_A_SHAPE_32x2
#endif

#if defined(DATA_A_IQ5_K)
void prepare_a_k32_iter(uint ib_k, uint8_t ib32, out int32_t qs_a[8], out vec2 a_scales) {
    const uint ib32_u = uint(ib32);
    const uint qs_idx = 8 * (ib32_u / 2);
    const uint qh_idx = 0;
    const uint table_offset0 = 32 * ((data_a[ib_k].extra >> (2 * ib32_u)) & uint16_t(1));
    const uint table_offset1 = 32 * ((data_a[ib_k].extra >> (2 * ib32_u + 1)) & uint16_t(1));

    qs_a[0] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx    ], data_a_packed32[ib_k].qh[qh_idx    ], ib32_u, table_offset0);
    qs_a[1] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 1], data_a_packed32[ib_k].qh[qh_idx + 1], ib32_u, table_offset0);
    qs_a[2] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 2], data_a_packed32[ib_k].qh[qh_idx + 2], ib32_u, table_offset0);
    qs_a[3] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 3], data_a_packed32[ib_k].qh[qh_idx + 3], ib32_u, table_offset0);
    qs_a[4] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 4], data_a_packed32[ib_k].qh[qh_idx + 4], ib32_u, table_offset1);
    qs_a[5] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 5], data_a_packed32[ib_k].qh[qh_idx + 5], ib32_u, table_offset1);
    qs_a[6] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 6], data_a_packed32[ib_k].qh[qh_idx + 6], ib32_u, table_offset1);
    qs_a[7] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 7], data_a_packed32[ib_k].qh[qh_idx + 7], ib32_u, table_offset1);
    const uint scales_h = uint(data_a[ib_k].scales_h[ib32_u / 2]) >> (4 * (ib32_u % 2));
    const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
    const int32_t scale0 = int32_t((scales_l & 0x0F) | ((scales_h << 4) & 0x30)) - 32;
    const int32_t scale1 = int32_t((scales_l >> 4) | ((scales_h << 2) & 0x30)) - 32;
    const float d = float(data_a[ib_k].d);
    a_scales = vec2(d * float(scale0), d * float(scale1));
}

void prepare_a_k32x2_iter_shifted(uint ib_k, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t laneid = uint16_t(gl_SubgroupInvocationID);
    const uint16_t row = laneid & uint16_t(3);
    const uint16_t subid = laneid >> 2u;
    const uint16_t ib32 = subid << 1u;
    const uint16_t qs_idx = ib32 << 2u;
    const uint16_t qh_idx = uint16_t(0);
    const uint16_t qs_idx0 = qs_idx + ((row & uint16_t(1)) << 2);
    const uint16_t qs_idx1 = qs_idx + (((row ^ uint16_t(1)) & uint16_t(1)) << 2);
    const uint16_t qh_idx0 = qh_idx + ((row & uint16_t(1)) << 2);
    const uint16_t qh_idx1 = qh_idx + (((row ^ uint16_t(1)) & uint16_t(1)) << 2);
    const uvec4 qs0 = uvec4(data_a_packed32[ib_k].qs[qs_idx0 + 0u],
                            data_a_packed32[ib_k].qs[qs_idx0 + 1u],
                            data_a_packed32[ib_k].qs[qs_idx0 + 2u],
                            data_a_packed32[ib_k].qs[qs_idx0 + 3u]);
    const uvec4 qs1 = uvec4(data_a_packed32[ib_k].qs[qs_idx1 + 0u],
                            data_a_packed32[ib_k].qs[qs_idx1 + 1u],
                            data_a_packed32[ib_k].qs[qs_idx1 + 2u],
                            data_a_packed32[ib_k].qs[qs_idx1 + 3u]);
    const uvec4 qh0 = uvec4(data_a_packed32[ib_k].qh[qh_idx0 + 0u],
                            data_a_packed32[ib_k].qh[qh_idx0 + 1u],
                            data_a_packed32[ib_k].qh[qh_idx0 + 2u],
                            data_a_packed32[ib_k].qh[qh_idx0 + 3u]);
    const uvec4 qh1 = uvec4(data_a_packed32[ib_k].qh[qh_idx1 + 0u],
                            data_a_packed32[ib_k].qh[qh_idx1 + 1u],
                            data_a_packed32[ib_k].qh[qh_idx1 + 2u],
                            data_a_packed32[ib_k].qh[qh_idx1 + 3u]);
    const uint16_t block0 = ib32 + ((row & uint16_t(2)) >> 1);
    const uint16_t block1 = ib32 + (((row ^ uint16_t(2)) & uint16_t(2)) >> 1);
    const uint16_t table_offset_bits = data_a[ib_k].extra >> (ib32 << 1);
    const uint table_offset0 = ((table_offset_bits >> row) & 1u) << 5u;
    const uint table_offset1 = ((table_offset_bits >> (row ^ uint16_t(1))) & 1u) << 5u;
    const uint table_offset2 = ((table_offset_bits >> (row ^ uint16_t(2))) & 1u) << 5u;
    const uint table_offset3 = ((table_offset_bits >> (row ^ uint16_t(3))) & 1u) << 5u;

    qs_a[0] = i32vec4(unpack_iq5_k(qs0.x, qh0.x, block0, table_offset0),
                      unpack_iq5_k(qs0.y, qh0.y, block0, table_offset0),
                      unpack_iq5_k(qs0.z, qh0.z, block0, table_offset0),
                      unpack_iq5_k(qs0.w, qh0.w, block0, table_offset0));
    qs_a[1] = i32vec4(unpack_iq5_k(qs1.x, qh1.x, block0, table_offset1),
                      unpack_iq5_k(qs1.y, qh1.y, block0, table_offset1),
                      unpack_iq5_k(qs1.z, qh1.z, block0, table_offset1),
                      unpack_iq5_k(qs1.w, qh1.w, block0, table_offset1));
    qs_a[2] = i32vec4(unpack_iq5_k(qs0.x, qh0.x, block1, table_offset2),
                      unpack_iq5_k(qs0.y, qh0.y, block1, table_offset2),
                      unpack_iq5_k(qs0.z, qh0.z, block1, table_offset2),
                      unpack_iq5_k(qs0.w, qh0.w, block1, table_offset2));
    qs_a[3] = i32vec4(unpack_iq5_k(qs1.x, qh1.x, block1, table_offset3),
                      unpack_iq5_k(qs1.y, qh1.y, block1, table_offset3),
                      unpack_iq5_k(qs1.z, qh1.z, block1, table_offset3),
                      unpack_iq5_k(qs1.w, qh1.w, block1, table_offset3));

    const uint16_t scale_l = uint16_t(data_a_packed32[ib_k].scales_l[subid >> 1] >>
                                      ((subid & uint16_t(1)) << 4));
    const uint scale_lo = u4x4_to_u8x4(scale_l);
    const uint scale_hi = u2x4_to_u8x4_sub32(data_a[ib_k].scales_h[subid]);
    a_scales = unpack8(int32_t(scale_lo | scale_hi));
}
#define SG_A_SHAPE_32x2
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

void prepare_a_k32_iter(uint ib_k, uint8_t ib32, out int32_t qs_a[8], out vec2 a_scales) {
    const uint ib32_u = uint(ib32);
    const uint qs_idx = 8 * (ib32_u / 2);
    const uint qh_idx = 8 * (ib32_u / 4);
    const uint shift = 2 * (ib32_u % 4);
    const uint table_offset0 = 64 * ((data_a[ib_k].extra >> (2 * ib32_u)) & uint16_t(1));
    const uint table_offset1 = 64 * ((data_a[ib_k].extra >> (2 * ib32_u + 1)) & uint16_t(1));

    qs_a[0] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx    ], data_a_packed32[ib_k].qh[qh_idx    ], shift, table_offset0);
    qs_a[1] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 1], data_a_packed32[ib_k].qh[qh_idx + 1], shift, table_offset0);
    qs_a[2] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 2], data_a_packed32[ib_k].qh[qh_idx + 2], shift, table_offset0);
    qs_a[3] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 3], data_a_packed32[ib_k].qh[qh_idx + 3], shift, table_offset0);
    qs_a[4] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 4], data_a_packed32[ib_k].qh[qh_idx + 4], shift, table_offset1);
    qs_a[5] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 5], data_a_packed32[ib_k].qh[qh_idx + 5], shift, table_offset1);
    qs_a[6] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 6], data_a_packed32[ib_k].qh[qh_idx + 6], shift, table_offset1);
    qs_a[7] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 7], data_a_packed32[ib_k].qh[qh_idx + 7], shift, table_offset1);
    const float d = float(data_a[ib_k].d);
    a_scales = vec2(d * float(data_a[ib_k].scales[2 * ib32_u]),
                    d * float(data_a[ib_k].scales[2 * ib32_u + 1]));
}

void prepare_a_k32x2_iter_shifted(uint ib_k, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t laneid = uint16_t(gl_SubgroupInvocationID);
    const uint16_t row = laneid & uint16_t(3);
    const uint16_t subid = laneid >> 2;
    const uint16_t ib32 = subid << 1;
    const uint16_t qs_idx = ib32 << 2;
    const uint16_t qh_idx = (ib32 >> 2) << 3;
    const uint16_t qs_idx0 = qs_idx + ((row & uint16_t(1)) << 2);
    const uint16_t qs_idx1 = qs_idx + (((row ^ uint16_t(1)) & uint16_t(1)) << 2);
    const uint16_t qh_idx0 = qh_idx + ((row & uint16_t(1)) << 2);
    const uint16_t qh_idx1 = qh_idx + (((row ^ uint16_t(1)) & uint16_t(1)) << 2);
    const uvec4 qs0 = uvec4(data_a_packed32[ib_k].qs[qs_idx0 + 0u],
                            data_a_packed32[ib_k].qs[qs_idx0 + 1u],
                            data_a_packed32[ib_k].qs[qs_idx0 + 2u],
                            data_a_packed32[ib_k].qs[qs_idx0 + 3u]);
    const uvec4 qs1 = uvec4(data_a_packed32[ib_k].qs[qs_idx1 + 0u],
                            data_a_packed32[ib_k].qs[qs_idx1 + 1u],
                            data_a_packed32[ib_k].qs[qs_idx1 + 2u],
                            data_a_packed32[ib_k].qs[qs_idx1 + 3u]);
    const uvec4 qh0 = uvec4(data_a_packed32[ib_k].qh[qh_idx0 + 0u],
                            data_a_packed32[ib_k].qh[qh_idx0 + 1u],
                            data_a_packed32[ib_k].qh[qh_idx0 + 2u],
                            data_a_packed32[ib_k].qh[qh_idx0 + 3u]);
    const uvec4 qh1 = uvec4(data_a_packed32[ib_k].qh[qh_idx1 + 0u],
                            data_a_packed32[ib_k].qh[qh_idx1 + 1u],
                            data_a_packed32[ib_k].qh[qh_idx1 + 2u],
                            data_a_packed32[ib_k].qh[qh_idx1 + 3u]);
    const uint16_t block0 = ib32 + ((row & uint16_t(2)) >> 1);
    const uint16_t block1 = ib32 + (((row ^ uint16_t(2)) & uint16_t(2)) >> 1);
    const uint16_t shift0 = (block0 & uint16_t(3)) << 1;
    const uint16_t shift1 = (block1 & uint16_t(3)) << 1;
    const uint16_t table_offset_bits = data_a[ib_k].extra >> (ib32 << 1);
    const uint table_offset0 = ((table_offset_bits >> row) & 1u) << 6u;
    const uint table_offset1 = ((table_offset_bits >> (row ^ uint16_t(1))) & 1u) << 6u;
    const uint table_offset2 = ((table_offset_bits >> (row ^ uint16_t(2))) & 1u) << 6u;
    const uint table_offset3 = ((table_offset_bits >> (row ^ uint16_t(3))) & 1u) << 6u;

    qs_a[0] = i32vec4(unpack_iq6_k(qs0.x, qh0.x, shift0, table_offset0),
                      unpack_iq6_k(qs0.y, qh0.y, shift0, table_offset0),
                      unpack_iq6_k(qs0.z, qh0.z, shift0, table_offset0),
                      unpack_iq6_k(qs0.w, qh0.w, shift0, table_offset0));
    qs_a[1] = i32vec4(unpack_iq6_k(qs1.x, qh1.x, shift0, table_offset1),
                      unpack_iq6_k(qs1.y, qh1.y, shift0, table_offset1),
                      unpack_iq6_k(qs1.z, qh1.z, shift0, table_offset1),
                      unpack_iq6_k(qs1.w, qh1.w, shift0, table_offset1));
    qs_a[2] = i32vec4(unpack_iq6_k(qs0.x, qh0.x, shift1, table_offset2),
                      unpack_iq6_k(qs0.y, qh0.y, shift1, table_offset2),
                      unpack_iq6_k(qs0.z, qh0.z, shift1, table_offset2),
                      unpack_iq6_k(qs0.w, qh0.w, shift1, table_offset2));
    qs_a[3] = i32vec4(unpack_iq6_k(qs1.x, qh1.x, shift1, table_offset3),
                      unpack_iq6_k(qs1.y, qh1.y, shift1, table_offset3),
                      unpack_iq6_k(qs1.z, qh1.z, shift1, table_offset3),
                      unpack_iq6_k(qs1.w, qh1.w, shift1, table_offset3));

    a_scales = unpack8(int32_t(data_a_packed32[ib_k].scales[subid]));
}
#define SG_A_SHAPE_32x2
#endif

#if defined(DATA_A_IQ2_K)
i32vec4 repack4(uint ib, uint8_t iqs) {
    const uint ib_k = ib / 8;
    const uint8_t ib32 = uint8_t(ib) & uint8_t(7);
    const uint8_t qs_idx = (bool(ib32 & uint8_t(4)) ? uint8_t(8) : uint8_t(0)) + (iqs << 2);
    const uint8_t ib32_shift = ib32 << 1;
    const uint8_t shift = ib32_shift & uint8_t(6);
    const bool is_hi_table = bool((data_a[ib_k].extra >> (ib32_shift + iqs)) & uint16_t(1));

    return i32vec4(unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx    ] >> shift) & 0x03030303, is_hi_table),
                   unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 1] >> shift) & 0x03030303, is_hi_table),
                   unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 2] >> shift) & 0x03030303, is_hi_table),
                   unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 3] >> shift) & 0x03030303, is_hi_table));
}

float get_d_scale(uint ib, uint8_t iqs) {
    const uint ib_k = ib / 8;
    const uint8_t ib32 = uint8_t(ib) & uint8_t(7);
    const uint8_t scale = data_a[ib_k].scales[ib32];
    return float(data_a[ib_k].d) * float(int32_t((scale >> (uint8_t(4) * iqs)) & uint8_t(0x0F)) - 8);
}

#endif

#if defined(DATA_A_IQ3_K)
i32vec4 repack4(uint ib, uint8_t iqs) {
    const uint ib_k = ib / 8;
    const uint8_t ib32 = uint8_t(ib) & uint8_t(7);
    const uint8_t qh_idx = iqs << 3;
    const uint8_t q_idx = (bool(ib32 & uint8_t(4)) ? uint8_t(16) : uint8_t(0)) + qh_idx;
    const bool is_hi_table = bool((data_a[ib_k].extra >> ((ib32 << 1) + iqs)) & uint16_t(1));

    i32vec4 result;
    [[unroll]] for (uint j = 0; j < 4; ++j) {
        const uint ql = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx + 2 * j], data_a_packed16[ib_k].qs[q_idx + 2 * j + 1]));
        const uint qh = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_idx + 2 * j], data_a_packed16[ib_k].qh[qh_idx + 2 * j + 1]));
        result[j] = unpack_iq3_k(ql, qh, uint16_t(ib32), is_hi_table);
    }
    return result;
}

float get_d_scale(uint ib, uint8_t iqs) {
    const uint ib_k = ib / 8;
    const uint8_t ib32 = uint8_t(ib) & uint8_t(7);
    const uint8_t scale_l = (data_a[ib_k].scales_l[ib32] >> (uint8_t(4) * iqs)) & uint8_t(0x0F);
    const bool sign = bool((data_a[ib_k].scales_h >> ((ib32 << 1) + iqs)) & uint16_t(1));
    return float(data_a[ib_k].d) * float(2 * scale_l + 1) * (sign ? -1.0 : 1.0);
}

#endif

#if defined(DATA_A_IQ4_K)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint qs_idx = 4 * ib32;
    const bool is_hi_table = bool((data_a[ib_k].extra >> (2 * ib32 + iqs)) & uint16_t(1));
    const uint shift = 4 * iqs;

    return i32vec4(unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx    ], shift, is_hi_table),
                   unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 1], shift, is_hi_table),
                   unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 2], shift, is_hi_table),
                   unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 3], shift, is_hi_table));
}

float get_d_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint scales_h = uint(data_a[ib_k].scales_h[ib32 / 2]) >> (4 * (ib32 % 2));
    const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
    const int32_t scale = iqs == 0 ? int32_t((scales_l & 0x0F) | ((scales_h << 4) & 0x30)) - 32
                                   : int32_t((scales_l >> 4) | ((scales_h << 2) & 0x30)) - 32;
    return float(data_a[ib_k].d) * float(scale);
}

#endif

#if defined(DATA_A_IQ5_K)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint qs_idx = 8 * (ib32 / 2) + 4 * iqs;
    const uint qh_idx = 4 * iqs;
    const uint table_offset = 32 * ((data_a[ib_k].extra >> (2 * ib32 + iqs)) & uint16_t(1));

    return i32vec4(unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx    ], data_a_packed32[ib_k].qh[qh_idx    ], ib32, table_offset),
                   unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 1], data_a_packed32[ib_k].qh[qh_idx + 1], ib32, table_offset),
                   unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 2], data_a_packed32[ib_k].qh[qh_idx + 2], ib32, table_offset),
                   unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 3], data_a_packed32[ib_k].qh[qh_idx + 3], ib32, table_offset));
}

float get_d_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint scales_h = uint(data_a[ib_k].scales_h[ib32 / 2]) >> (4 * (ib32 % 2));
    const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
    const int32_t scale = iqs == 0 ? int32_t((scales_l & 0x0F) | ((scales_h << 4) & 0x30)) - 32
                                   : int32_t((scales_l >> 4) | ((scales_h << 2) & 0x30)) - 32;
    return float(data_a[ib_k].d) * float(scale);
}

#endif

#if defined(DATA_A_IQ6_K)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint qs_idx = 8 * (ib32 / 2) + 4 * iqs;
    const uint qh_idx = 8 * (ib32 / 4) + 4 * iqs;
    const uint shift = 2 * (ib32 % 4);
    const uint table_offset = 64 * ((data_a[ib_k].extra >> (2 * ib32 + iqs)) & uint16_t(1));

    return i32vec4(unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx    ], data_a_packed32[ib_k].qh[qh_idx    ], shift, table_offset),
                   unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 1], data_a_packed32[ib_k].qh[qh_idx + 1], shift, table_offset),
                   unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 2], data_a_packed32[ib_k].qh[qh_idx + 2], shift, table_offset),
                   unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 3], data_a_packed32[ib_k].qh[qh_idx + 3], shift, table_offset));
}

float get_d_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    return float(data_a[ib_k].d) * float(data_a[ib_k].scales[2 * ib32 + iqs]);
}

#endif


#if defined(SG_IQK)

#define SG_SCATTER_I32_STEP 8u

uint16_t get_ib32_scatter_offset() {
    const uint lane = gl_SubgroupInvocationID;
#if SUBGROUP_SIZE == 16
    return uint16_t((lane & 8u) >> 1u);
#elif SUBGROUP_SIZE == 32
    return uint16_t((lane & 8u) >> 1u) + uint16_t((lane & 0x10u) >> 3u);
#elif SUBGROUP_SIZE == 64
    return uint16_t((lane & 8u) >> 1u) + uint16_t((lane & 0x30u) >> 4u);
#else
#error unsupported subgroup size
#endif
}

uint16_t get_i32_scatter_offset() {
    return uint16_t(SG_SCATTER_I32_STEP * get_ib32_scatter_offset() + (gl_SubgroupInvocationID & 7u));
}

#if defined(DATA_A_IQ2_K)
#define SG_IQK_LANES_PER_32       (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_HALF        (QUANT_K_Q8_1 / 4)

void prepare_a_sg(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const uint16_t extra_bit_idx = uint16_t(lane / (SUBGROUP_SIZE / 16u));
    const uint qs_u32 = data_a_packed32[ib_k].qs[lane % uint16_t(16)];

    const uint16_t q_offset = uint16_t((lane * SG_QS_PER_ITER) % SG_IQK_QS_PER_HALF);
    const uint16_t qs_base = lane >= (SUBGROUP_SIZE / 2u) ? uint16_t(SG_IQK_QS_PER_HALF) : uint16_t(0);
    const uint16_t shift_l = uint16_t(2u * (ib32 % 4u));
    const bool is_hi_table = bool((data_a[ib_k].extra >> extra_bit_idx) & uint16_t(1));

    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q32 = q_offset + q;
        const uint q2 = subgroupShuffle(qs_u32, qs_base + q32);
        qs_a[q] = unpack_iq2_k((q2 >> shift_l) & 0x03030303u, is_hi_table);
    }

    const uint8_t scale_l = data_a[ib_k].scales[ib32];
    const bool scale_high_half = bool(lane & (SG_IQK_LANES_PER_32 / 2u));
    a_scale = float(data_a[ib_k].d) * float(int32_t((scale_l >> (scale_high_half ? 4 : 0)) & uint8_t(0x0F)) - 8);
}

void prepare_a_sg_scatter(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out int8_t qscale_a[SG_QS_PER_ITER]) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t extra = data_a[ib_k].extra;
    const uint qs_u32 = data_a_packed32[ib_k].qs[lane & uint16_t(15)];
    const uint scale_u32 = data_a_packed32[ib_k].scales[(lane & 15u) / 8u];
    const uint ib32_offset = get_ib32_scatter_offset();
#if SUBGROUP_SIZE == 16
    const uint16_t ib32_in_group_offset = uint16_t(0);
    const uint16_t qs_base_shift = uint16_t(0);
#else
    const uint16_t ib32_in_group_offset = uint16_t(lane / 16u) * uint16_t(64u / SUBGROUP_SIZE);
    const uint16_t qs_base_shift = uint16_t(lane / 16u) * uint16_t((64u / SUBGROUP_SIZE) * 2);
#endif
    const bool is_high_subblock = bool(lane & 4u);
    const uint32_t scale_shifted = (scale_u32 >> (uint16_t(is_high_subblock ? 4u : 0u) + uint16_t(8) * ib32_in_group_offset)) & 0x0F0F0F0Fu;

    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint ib32 = ib32_offset + q;
        const uint ib32_in_group = ib32_in_group_offset + q;
        const uint16_t shift_l = qs_base_shift + uint16_t(2u * q);
        const uint extra_bit_idx = 2u * ib32 + (is_high_subblock ? 1u : 0u);
        const bool is_hi_table = bool((extra >> extra_bit_idx) & 1u);
        const uint8_t scale = unpack8(scale_shifted)[q];

        qs_a[q] = unpack_iq2_k((qs_u32 >> shift_l) & 0x03030303u, is_hi_table);
        qscale_a[q] = int8_t(scale) - int8_t(8);
    }
}

void prepare_a_sg_scatter2(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out int8_t qscale_a) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t extra = data_a[ib_k].extra;
    const uint qs_u32 = data_a_packed32[ib_k].qs[lane & uint16_t(15)];
    const uint scale_u32 = data_a_packed32[ib_k].scales[(lane & 15u) / 8u];
    const uint ib32_offset = get_ib32_scatter_offset();
#if SUBGROUP_SIZE == 16
    const uint16_t ib32_in_group_offset = uint16_t(0);
    const uint16_t qs_base_shift = uint16_t(0);
#else
    const uint16_t ib32_in_group_offset = uint16_t(lane / 16u) * uint16_t(64u / SUBGROUP_SIZE);
    const uint16_t qs_base_shift = uint16_t(lane / 16u) * uint16_t((64u / SUBGROUP_SIZE) * 2);
#endif
    const bool is_high_subblock = bool(lane & 4u);
    const uint32_t scale_shifted = (scale_u32 >> (uint16_t(is_high_subblock ? 4u : 0u) + uint16_t(8) * ib32_in_group_offset)) & 0x0F0F0F0Fu;

    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint ib32 = ib32_offset + q;
        const uint ib32_in_group = ib32_in_group_offset + q;
        const uint16_t shift_l = qs_base_shift + uint16_t(2u * q);
        const uint extra_bit_idx = 2u * ib32 + (is_high_subblock ? 1u : 0u);
        const bool is_hi_table = bool((extra >> extra_bit_idx) & 1u);
        qs_a[q] = unpack_iq2_k((qs_u32 >> shift_l) & 0x03030303u, is_hi_table);
    }

#if SUBGROUP_SIZE == 16
    const uint small_block = ((uint(lane) & 3u) << 1u) |
                             ((uint(lane) & 4u) >> 2u) |
                             (uint(lane) & 8u);
#else
    const uint small_block = uint(lane) % 16u;
#endif
    const uint ib32 = small_block / 2u;
    const bool scale_high_half = bool(small_block & 1u);
    const uint8_t scale_l = data_a[ib_k].scales[ib32];
    qscale_a = int8_t((scale_l >> (scale_high_half ? 4u : 0u)) & uint8_t(0x0F)) - int8_t(8);
}
//#define SG_SUB_SCALE 1
//#define SG_SCATTER_K 1
#endif

#if defined(DATA_A_IQ3_K)
#define SG_IQK_LANES_PER_32      (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_HALF       (QUANT_K_Q8_1 / 4)

// A 32-element IQ3K sub-block is split across SG_IQK_LANES_PER_32 lanes.
// Each lane owns SG_K elements and SG_QS_PER_ITER packed int32 values.
void prepare_a_sg(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const uint16_t extra_bit_idx = uint16_t(lane / (SUBGROUP_SIZE / 16u));

    const uint qs_offset = 2u * (lane % uint16_t(16));
    const uint qs_u32 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_offset],
                                       data_a_packed16[ib_k].qs[qs_offset + 1u]));

    const uint qh_offset = 2u * (lane % uint16_t(8));
    const uint qh_u32 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_offset],
                                       data_a_packed16[ib_k].qh[qh_offset + 1u]));

    const uint16_t q_offset = uint16_t((lane * SG_QS_PER_ITER) % SG_IQK_QS_PER_HALF);
    const uint16_t qs_base = lane >= (SUBGROUP_SIZE / 2u) ? uint16_t(SG_IQK_QS_PER_HALF) : uint16_t(0);
    const uint16_t shift_h = uint16_t(ib32);
    const uint16_t shift_l = uint16_t(2u * (ib32 % 4u));
    const uint table_offset = ((data_a[ib_k].extra >> extra_bit_idx) & uint16_t(1)) << 6;

    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q32 = q_offset + q;
        const uint ql = subgroupShuffle(qs_u32, qs_base + q32);
        const uint qh = subgroupShuffle(qh_u32, q32);
        const uint indexes = ((ql >> shift_l) & 0x03030303u) |
                             (((qh >> shift_h) & 0x01010101u) << 2u);
        const uint index0 = table_offset + dotPacked4x8EXT(indexes & 0xffffu, 0x00000801u);
        const uint index1 = table_offset + dotPacked4x8EXT(indexes, 0x08010000u);
        qs_a[q] = int32_t(pack32(u16vec2(iq3k_table[index0], iq3k_table[index1])));
    }

    const uint16_t scale_l = uint16_t(data_a[ib_k].scales_l[ib32]);
    const bool scale_high_half = bool(lane & (SG_IQK_LANES_PER_32 / 2u));
    const uint16_t scale_l_shifted = scale_l >> (scale_high_half ? 4 : 0);
    const int16_t scale = int16_t((scale_l_shifted & uint16_t(0x0F)) * uint16_t(2) + uint16_t(1));
    const bool sign = bool((data_a[ib_k].scales_h >> extra_bit_idx) & uint16_t(1));
    a_scale = float(data_a[ib_k].d) * float(sign ? -scale : scale);
}

void prepare_a_sg_scatter(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out int8_t qscale_a[SG_QS_PER_ITER]) {
    const uint lane = gl_SubgroupInvocationID;
    const uint qs_offset = 2u * (lane & 15u);
    const uint qs_u32 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_offset],
                                       data_a_packed16[ib_k].qs[qs_offset + 1u]));
    const uint qh_offset = 2u * (lane & 7u);
    const uint qh_u32 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_offset],
                                       data_a_packed16[ib_k].qh[qh_offset + 1u]));
    const uint scale_u32 = pack32(u16vec2(data_a_packed16[ib_k].scales_l[2u * ((lane & 15u) / 8u)],
                                          data_a_packed16[ib_k].scales_l[2u * ((lane & 15u) / 8u) + 1u]));
    const uint scales_h = uint(data_a_packed16[ib_k].scales_h);
    const uint16_t extra = data_a[ib_k].extra;
    const uint ib32_offset = get_ib32_scatter_offset();
    const uint ib32_in_group_offset = (lane / 16u) * (64u / SUBGROUP_SIZE);
    const bool is_high_subblock = bool(lane & 4u);

    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint ib32 = ib32_offset + q;
        const uint ib32_in_group = ib32_in_group_offset + q;
        const uint shift_l = 2u * ib32_in_group;
        const uint shift_h = ib32;
        const uint bit_idx = 2u * ib32 + (is_high_subblock ? 1u : 0u);
        const uint indexes = ((qs_u32 >> shift_l) & 0x03030303u) |
                             (((qh_u32 >> shift_h) & 0x01010101u) << 2u);
        const uint table_offset = ((extra >> bit_idx) & uint16_t(1)) << 6u;
        const uint index0 = table_offset + dotPacked4x8EXT(indexes & 0xffffu, 0x00000801u);
        const uint index1 = table_offset + dotPacked4x8EXT(indexes, 0x08010000u);
        const uint16_t scale = uint16_t(scale_u32 >> (8u * ib32_in_group + (is_high_subblock ? 4u : 0u))) & uint16_t(0x0Fu);
        const int8_t scale_shifted = int8_t(scale << 1 | uint8_t(1));
        const bool sign = bool((scales_h >> bit_idx) & 1u);

        qs_a[q] = int32_t(pack32(u16vec2(iq3k_table[index0], iq3k_table[index1])));
        qscale_a[q] = sign ? -scale_shifted : scale_shifted;
    }
}

void prepare_a_sg_scatter2(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out int8_t qscale_a) {
    const uint lane = gl_SubgroupInvocationID;
    const uint qs_offset = 2u * (lane & 15u);
    const uint qs_u32 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_offset],
                                       data_a_packed16[ib_k].qs[qs_offset + 1u]));
    const uint qh_offset = 2u * (lane & 7u);
    const uint qh_u32 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_offset],
                                       data_a_packed16[ib_k].qh[qh_offset + 1u]));
    const uint ib32_offset = get_ib32_scatter_offset();
    const uint ib32_in_group_offset = (lane / 16u) * (64u / SUBGROUP_SIZE);
    const bool is_high_subblock = bool(lane & 4u);

    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint ib32 = ib32_offset + q;
        const uint ib32_in_group = ib32_in_group_offset + q;
        const uint shift_l = 2u * ib32_in_group;
        const uint shift_h = ib32;
        const uint bit_idx = 2u * ib32 + (is_high_subblock ? 1u : 0u);
        const uint indexes = ((qs_u32 >> shift_l) & 0x03030303u) |
                             (((qh_u32 >> shift_h) & 0x01010101u) << 2u);
        const uint table_offset = ((data_a[ib_k].extra >> bit_idx) & uint16_t(1)) << 6u;
        const uint index0 = table_offset + dotPacked4x8EXT(indexes & 0xffffu, 0x00000801u);
        const uint index1 = table_offset + dotPacked4x8EXT(indexes, 0x08010000u);

        qs_a[q] = int32_t(pack32(u16vec2(iq3k_table[index0], iq3k_table[index1])));
    }

    const uint small_block = lane % 16u;
    const uint ib32 = small_block / 2u;
    const bool scale_high_half = bool(small_block & 1u);
    const uint8_t scale_l = data_a[ib_k].scales_l[ib32];
    const uint scale = (uint(scale_l) >> (scale_high_half ? 4u : 0u)) & 0x0Fu;
    const uint bit_idx = 2u * ib32 + (scale_high_half ? 1u : 0u);
    const int8_t scale_shifted = int8_t(scale << 1u | uint8_t(1));
    const bool sign = bool((uint(data_a[ib_k].scales_h) >> bit_idx) & 1u);
    qscale_a = sign ? -scale_shifted : scale_shifted;
}

//#define SG_SCATTER_K 1
#endif

#if defined(DATA_A_IQ4_K)
#define SG_IQK_LANES_PER_32      (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_32         (QUANT_K_Q8_1 / 8)

void prepare_a_sg(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const bool scale_high_half = bool(lane & uint16_t(SG_IQK_LANES_PER_32 / 2u));
    const uint16_t q_offset = uint16_t((uint(lane) * SG_QS_PER_ITER) % SG_IQK_QS_PER_32);

#if SUBGROUP_SIZE == 16
    const uvec2 qs_u32 = uvec2(data_a_packed32[ib_k].qs[uint(lane) * 2u],
                               data_a_packed32[ib_k].qs[uint(lane) * 2u + 1u]);
#else
    const uint qs_u32 = data_a_packed32[ib_k].qs[uint(lane) & 31u];
#endif

    const uint extra_bit_idx = 2u * uint(ib32) + (scale_high_half ? 1u : 0u);
    const bool is_hi_table = bool((data_a[ib_k].extra >> extra_bit_idx) & uint16_t(1));
    const uint shift = scale_high_half ? 4u : 0u;

#if SUBGROUP_SIZE == 16
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; q += 2u) {
        const uint q_idx = 4u * uint(ib32) + uint(q_offset) + q;
        const uvec2 qs_pair = subgroupShuffle(qs_u32, q_idx >> 1u);
        qs_a[q] = unpack_iq4_k(qs_pair.x, shift, is_hi_table);
        qs_a[q + 1u] = unpack_iq4_k(qs_pair.y, shift, is_hi_table);
    }
#else
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q_idx = 4u * uint(ib32) + uint(q_offset) + q;
        const uint q4 = subgroupShuffle(qs_u32, q_idx);
        qs_a[q] = unpack_iq4_k(q4, shift, is_hi_table);
    }
#endif

    const uint scales_h = uint(data_a[ib_k].scales_h[uint(ib32) / 2u]) >> (4u * (uint(ib32) % 2u));
    const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
    const uint scale = scale_high_half
        ? ((scales_l >> 4u) | ((scales_h << 2u) & 0x30u))
        : ((scales_l & 0x0Fu) | ((scales_h << 4u) & 0x30u));
    a_scale = float(data_a[ib_k].d) * float(int32_t(scale) - 32);
}
#endif

#if defined(DATA_A_IQ5_K)
#define SG_IQK_LANES_PER_32      (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_32         (QUANT_K_Q8_1 / 4)

void prepare_a_sg(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const bool scale_high_half = bool(lane & uint16_t(SG_IQK_LANES_PER_32 / 2u));
    const uint16_t q_offset = uint16_t((uint(lane) * SG_QS_PER_ITER) % SG_IQK_QS_PER_32);

#if SUBGROUP_SIZE == 16
    const uvec2 qs_u32 = uvec2(data_a_packed32[ib_k].qs[uint(lane) * 2u],
                               data_a_packed32[ib_k].qs[uint(lane) * 2u + 1u]);
#else
    const uint qs_u32 = data_a_packed32[ib_k].qs[uint(lane) & 31u];
#endif
    const uint qh_u32 = data_a_packed32[ib_k].qh[uint(lane) & 7u];

    const uint extra_bit_idx = 2u * uint(ib32) + (scale_high_half ? 1u : 0u);
    const uint table_offset = ((data_a[ib_k].extra >> extra_bit_idx) & uint16_t(1)) << 5u;

#if SUBGROUP_SIZE == 16
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; q += 2u) {
        const uint q_idx = 8u * (uint(ib32) / 2u) + uint(q_offset) + q;
        const uvec2 qs_pair = subgroupShuffle(qs_u32, q_idx >> 1u);
        const uint qh0 = subgroupShuffle(qh_u32, uint(q_offset) + q);
        const uint qh1 = subgroupShuffle(qh_u32, uint(q_offset) + q + 1u);
        qs_a[q] = unpack_iq5_k(qs_pair.x, qh0, uint(ib32), table_offset);
        qs_a[q + 1u] = unpack_iq5_k(qs_pair.y, qh1, uint(ib32), table_offset);
    }
#else
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q_idx = 8u * (uint(ib32) / 2u) + uint(q_offset) + q;
        const uint qs = subgroupShuffle(qs_u32, q_idx);
        const uint qh = subgroupShuffle(qh_u32, uint(q_offset) + q);
        qs_a[q] = unpack_iq5_k(qs, qh, uint(ib32), table_offset);
    }
#endif

    const uint scales_h = uint(data_a[ib_k].scales_h[uint(ib32) / 2u]) >> (4u * (uint(ib32) % 2u));
    const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
    const uint scale = scale_high_half
        ? ((scales_l >> 4u) | ((scales_h << 2u) & 0x30u))
        : ((scales_l & 0x0Fu) | ((scales_h << 4u) & 0x30u));
    a_scale = float(data_a[ib_k].d) * float(int32_t(scale) - 32);
}
#endif

#if defined(DATA_A_IQ6_K)
#define SG_IQK_LANES_PER_32      (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_32         (QUANT_K_Q8_1 / 4)

void prepare_a_sg(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const bool scale_high_half = bool(lane & uint16_t(SG_IQK_LANES_PER_32 / 2u));
    const uint16_t q_offset = uint16_t((uint(lane) * SG_QS_PER_ITER) % SG_IQK_QS_PER_32);

#if SUBGROUP_SIZE == 16
    const uvec2 qs_u32 = uvec2(data_a_packed32[ib_k].qs[uint(lane) * 2u],
                               data_a_packed32[ib_k].qs[uint(lane) * 2u + 1u]);
#else
    const uint qs_u32 = data_a_packed32[ib_k].qs[uint(lane) & 31u];
#endif
    const uint qh_u32 = data_a_packed32[ib_k].qh[uint(lane) & 15u];

    const uint extra_bit_idx = 2u * uint(ib32) + (scale_high_half ? 1u : 0u);
    const uint table_offset = ((data_a[ib_k].extra >> extra_bit_idx) & uint16_t(1)) << 6u;
    const uint shift = 2u * (uint(ib32) % 4u);

#if SUBGROUP_SIZE == 16
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; q += 2u) {
        const uint q_idx = 8u * (uint(ib32) / 2u) + uint(q_offset) + q;
        const uvec2 qs_pair = subgroupShuffle(qs_u32, q_idx >> 1u);
        const uint qh0 = subgroupShuffle(qh_u32, 8u * (uint(ib32) / 4u) + uint(q_offset) + q);
        const uint qh1 = subgroupShuffle(qh_u32, 8u * (uint(ib32) / 4u) + uint(q_offset) + q + 1u);
        qs_a[q] = unpack_iq6_k(qs_pair.x, qh0, shift, table_offset);
        qs_a[q + 1u] = unpack_iq6_k(qs_pair.y, qh1, shift, table_offset);
    }
#else
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q_idx = 8u * (uint(ib32) / 2u) + uint(q_offset) + q;
        const uint qs = subgroupShuffle(qs_u32, q_idx);
        const uint qh = subgroupShuffle(qh_u32, 8u * (uint(ib32) / 4u) + uint(q_offset) + q);
        qs_a[q] = unpack_iq6_k(qs, qh, shift, table_offset);
    }
#endif

    const uint scale_idx = 2u * uint(ib32) + (scale_high_half ? 1u : 0u);
    a_scale = float(data_a[ib_k].d) * float(data_a[ib_k].scales[scale_idx]);
}
#endif

#endif


#if defined(DATA_A_IQK_ROW)
#extension GL_EXT_expect_assume : require
#if defined(DATA_A_IQ2_KS) || defined(DATA_A_IQ2_KL) || defined(DATA_A_IQ3_KS)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

uint8_t iqks_load_u8(uint offset) {
    return data_a[offset];
}

uint16_t iqks_load_u16(uint offset) {
#if defined(A_TYPE_PACKED16)
    assumeEXT((offset & 1u) == 0u);
    assumeEXT((offset & uint32_t(-2)) == offset);
    return data_a_packed16[offset / 2];
#else
    return pack8(u8vec2(iqks_load_u8(offset), iqks_load_u8(offset + 1)));
#endif
}

uint32_t iqks_load_u32(uint offset) {
#if defined(A_TYPE_PACKED32)
    assumeEXT((offset & 3u) == 0u);
    assumeEXT((offset & uint32_t(-4)) == offset);
    return data_a_packed32[offset / 4];
#else
    return pack32(u16vec2(iqks_load_u16(offset), iqks_load_u16(offset + 2)));
#endif
}

float iqks_row_scale(uint row_offset) {
#if defined(DATA_A_IQ2_KS) || defined(DATA_A_IQ2_KL) || defined(DATA_A_IQ3_KS)
    return float(uint16BitsToHalf(iqks_load_u16(row_offset)));
#else
    return uintBitsToFloat(iqks_load_u32(row_offset));
#endif
}

#if defined(DATA_A_IQ1_KT) || defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT) || defined(DATA_A_IQ4_KT)
int8_t iqkt_next(inout uint state) {
    state *= 0xCBAC1FEDu;
    const uint value = state & 0x3F3F3F3Fu;
    return int8_t(dotPacked4x8EXT(int32_t(value), int32_t(0x01010101u)) + (-126));
}

//#define IQKT_MULTI_TURN 1
#if defined(IQKT_MULTI_TURN)
#extension GL_KHR_shader_subgroup_basic : require
const uint32_t iqkt_turn_muler[8] = {
    0x1u, 0xCBAC1FEDu, 0xC8734169u, 0xBD2B4535u, 0xE50C7D11u, 0x1220D7BDu, 0x94839CF9u, 0x58267985u
};
uint32_t iqkt_multi_turn(uint32_t state, uint8_t turn) {
    const uint32_t muler = gl_SubgroupInvocationID < 8
        ? iqkt_turn_muler[gl_SubgroupInvocationID]
        : 0u;
    // return state * subgroupShuffle(muler, turn);
    return state * iqkt_turn_muler[turn];
}
#endif

#endif

int32_t iqks_value4(uint block_offset, uint8_t element) {
    const uint8_t pos = uint8_t(element % 32u);

#if defined(DATA_A_IQ1_KT) || defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT) || defined(DATA_A_IQ4_KT)
    const uint8_t group8 = uint8_t(element / 8u);
    const uint8_t pos8 = uint8_t(element % 8u);
    const uint8_t ib32 = uint8_t(element / 32u);
    uint state;

#if defined(DATA_A_IQ1_KT)
    const uint8_t sh = iqks_load_u8(block_offset + ib32);
    const uint8_t x = iqks_load_u8(block_offset + 8 + group8);
    const uint8_t y_ = iqks_load_u8(block_offset + 40 + group8 % 16);
    const uint8_t y = bool(element & 128u) ? y_ >> 4u : y_ & uint8_t(0x0F);
    const bool z = bool(sh & (uint8_t(1) << (4 + group8 % 4)));
    state = pack16(u8vec2(x, y)) | uint16_t(z ? 0x2000u : 0x1000u);
#elif defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT)
    state = uint(iqks_load_u16(block_offset + 4 + 2 * group8)) + 4096;
#else
    const uint header = iqks_load_u32(block_offset + 4 * ib32);
    //const uint8_t seed_index = uint16_t(2 * group8 + pos8 / 4u);
    const uint16_t seed_index = uint16_t(bitfieldExtract(uint32_t(element), 2, 1)) + ((element >> 2) & uint8_t(0x3e));
    const uint8_t x = iqks_load_u8(block_offset + 32 + seed_index);
    const uint8_t y_ = iqks_load_u8(block_offset + 96 + (seed_index % 32u));
    const uint8_t y = test_bit_to_get_high4(element, 7, y_);
    // const uint8_t z = uint8_t((header >> (8 + 3 * (seed_index % 8))) & 7);
    const uint32_t z = bitfieldExtract(header, 8 + 3 * (seed_index % uint16_t(8)), 3);
    // const bool w = bool(header & 1u);
    const uint32_t high = z + test_bit_to_get_bit(header, 0, 3) + 1u;
    state = pack16(u8vec2(x, y)) | (uint32_t(high) << 12u);
#endif

    const uint8_t count =
#if defined(DATA_A_IQ4_KT)
        uint8_t(pos8 % 4);
#else
        pos8;
#endif

#if defined(IQKT_MULTI_TURN)
    state = iqkt_multi_turn(state, count);
#else
    [[unroll]] for (uint8_t j = uint8_t(0); j < count; ++j) {
        iqkt_next(state);
    }
#endif

    i8vec4 values = i8vec4(iqkt_next(state), iqkt_next(state),
                           iqkt_next(state), iqkt_next(state));

#if defined(DATA_A_IQ3_KT)
    const u8vec4 signs = unpack8((iqks_load_u32(block_offset + 68 + element % 32) >> ib32) & 0x01010101u);
    const i8vec4 desired_sign = -i8vec4(signs);
    const i8vec4 sign_mask = (values >> int8_t(7)) ^ desired_sign;
    values = (values ^ sign_mask) - sign_mask;
#endif
    return pack32(values);
#elif defined(DATA_A_IQ4_KSS)
    const uint ib32 = element / 32;
    const uint word_offset = block_offset + 4 * (4 * ib32 + (pos % 16) / 4);
    uint values = iqks_load_u32(word_offset) & 0xFFFEFFFE;
    values ^= values >> 1;
    uint scale_bits = 0;
    [[unroll]] for (uint j = 0; j < 4; ++j) {
        scale_bits |= (iqks_load_u32(block_offset + 4 * (4 * ib32 + j)) & 0x00010001) << (2 * j);
    }
    const uint8_t scale = uint8_t(scale_bits | (scale_bits >> 15));
    const bool is_hi_table = bool(scale & uint8_t(1));
    return unpack_iq4_k(values, 4u * (pos / 16u), is_hi_table);
#elif defined(DATA_A_IQ2_KS)
    const bool high_half = bool(element & 128u);
    const uint group = (element % 128) / 32;
    const uint16_t extra = iqks_load_u16(block_offset) >> (high_half ? 4u : 0u);
    const uint values = iqks_load_u32(block_offset + 6 + (high_half ? 32u : 0u) + pos);
    const bool is_hi_table = bool(extra & (uint16_t(1) << group));
    return unpack_iq2_k((values >> (2 * group)) & 0x03030303u, is_hi_table);
#elif defined(DATA_A_IQ2_KL)
    const uint8_t ib64 = uint8_t(element / 64u);
    const uint8_t pos32 = uint8_t(element % 32u);
    const bool high_half = bool(element & uint8_t(32));
    const bool odd = bool(element & uint8_t(1));
    const uint16_t packed = iqks_load_u16(block_offset + 6 + 16 * ib64 + pos32 / 2);
    const uint16_t high = iqks_load_u16(block_offset + 70 + pos32 / 2);
    const uint high_shift = 2 * ib64 + (high_half ? 1u : 0u);
    const uint16_t index_even = ((high_half ? packed >> 3u : packed << 1u) & uint16_t(0x1E1E)) |
                                (((high >> high_shift) & uint16_t(0x0101)) << 5u);
    const uint16_t index = odd ? index_even + uint16_t(0x0101) : index_even;
    const u8vec2 indexes = unpack8(index);
    return int32_t(pack32(u16vec2(iq2kl_table[indexes.x], iq2kl_table[indexes.y])));
#elif defined(DATA_A_IQ3_KS)
    const uint8_t shift_h = element >> 5;
    const uint8_t group = shift_h & uint8_t(3);
    const bool high_half = bool(shift_h & uint8_t(4));
    //const uint16_t extra = iqks_load_u16(block_offset) >> (high_half ? 4u : 0u);
    const uint low = iqks_load_u32(block_offset + 6 + (high_half ? 32u : 0u) + pos);
    const uint high = iqks_load_u32(block_offset + 70 + pos);
    const uint16_t hi_table_bit_idx = uint16_t(shift_h) + uint16_t(8);
    const bool is_hi_table = bool(iqks_load_u16(block_offset) & (uint16_t(1) << hi_table_bit_idx));
    //const bool is_hi_table = bool(extra & (uint16_t(1) << (8 + group)));
    return unpack_iq3_k(low, high, uint16_t(shift_h), is_hi_table);
#elif defined(DATA_A_IQ4_KS)
    const uint ib32 = element / 32;
    const uint8_t scale = iqks_load_u8(block_offset + ib32);
    const uint values = iqks_load_u32(block_offset + 8 + 16 * ib32 + pos % 16);
    const bool is_hi_table = bool(scale & uint8_t(1));
    return unpack_iq4_k(values, 4u * (pos / 16u), is_hi_table);
#else
    const uint ib64 = element / 64;
    const uint pos64 = element % 64;
    const uint pos32 = pos64 % 32;
    const bool high_half = bool(pos64 & 32u);
    const uint8_t scale = iqks_load_u8(block_offset + 2 * ib64 + (high_half ? 1u : 0u));
    const uint values = iqks_load_u32(block_offset + 8 + 32 * ib64 + pos32);
    const uint high = iqks_load_u32(block_offset + 136 + pos32);
    const uint value_shift = high_half ? 4u : 0u;
    const uint high_shift = 2 * ib64 + (high_half ? 1u : 0u);
    const uint table_offset = bool(scale & uint8_t(1)) ? 32u : 0u;
    const uint packed = ((values >> value_shift) & 0x0F0F0F0Fu) |
                        (((high >> high_shift) & 0x01010101u) << 4);
    const u8vec4 indexes = unpack8(packed);
    return pack32(i8vec4(kvalues_iq5_k[indexes.x + table_offset],
                         kvalues_iq5_k[indexes.y + table_offset],
                         kvalues_iq5_k[indexes.z + table_offset],
                         kvalues_iq5_k[indexes.w + table_offset]));
#endif
}

// The caller splits each 256-element IQKS block into eight 32-element sub-blocks.
// ib32 identifies the selected sub-block and is always in the range 0..7.
float iqks_d_scale(float row_scale, uint block_offset, uint8_t ib32) {
#if defined(DATA_A_IQ1_KT)
    const uint8_t scale = iqks_load_u8(block_offset + ib32) & uint8_t(0x0F);
    return row_scale * float(kvalues_iqkt_scale[scale]);
#elif defined(DATA_A_IQ2_KT)
    const uint8_t group = ib32 & uint8_t(3);
    const uint8_t scale_ = iqks_load_u8(block_offset + group);
    const uint8_t scale = test_bit_to_get_high4(ib32, uint8_t(2), scale_);
    return row_scale * float(kvalues_iqkt_scale[scale]);
#elif defined(DATA_A_IQ3_KT)
    const uint8_t group = ib32 & uint8_t(3);
    const uint8_t scale_ = iqks_load_u8(block_offset + group);
    const uint8_t scale = test_bit_to_get_high4(ib32, uint8_t(2), scale_);
    return row_scale * float(scale);
#elif defined(DATA_A_IQ4_KT)
    return row_scale * float(int16_t(iqks_load_u8(block_offset + (ib32 << 2)) >> 1) - int16_t(64));
#elif defined(DATA_A_IQ4_KSS)
    const uint8_t ib32_offset = ib32 << 4;
    const uint scale_offset = block_offset + ib32_offset;
    const uint scale_bits =
        ( iqks_load_u32(scale_offset     ) & 0x00010001) |
        ((iqks_load_u32(scale_offset +  4) & 0x00010001) << 2) |
        ((iqks_load_u32(scale_offset +  8) & 0x00010001) << 4) |
        ((iqks_load_u32(scale_offset + 12) & 0x00010001) << 6);
    const uint8_t scale = uint8_t(scale_bits | (scale_bits >> 15));
    return row_scale * int8_t((scale & uint8_t(254)) - uint8_t(127));
#elif defined(DATA_A_IQ2_KS)
    const uint8_t packed_scale = iqks_load_u8(block_offset + 2 + (ib32 >> 1));
    const uint8_t scale_lo = test_bit_to_get_high4(ib32, 0, packed_scale);
    const uint16_t scale_hi_bit = uint16_t(1) << (uint16_t(ib32) + uint16_t(8));
    const uint16_t scale_hi = bool(iqks_load_u16(block_offset) & scale_hi_bit) ? uint16_t(0) : uint16_t(-16);
    const int16_t scale = int16_t(uint16_t(scale_lo) | scale_hi);
    return row_scale * float(scale);
#elif defined(DATA_A_IQ2_KL)
    const uint8_t group = ib32 & uint8_t(3);
    const uint8_t scale_ = iqks_load_u8(block_offset + 2 + group);
    //const uint scale_lo = test_bit_to_get_high4(ib32, 2, scale_);
    const uint scale_lo = bitfieldExtract(uint32_t(scale_), ib32 & 4, 4);
    const uint scale_hi = iqks_load_u16(block_offset) >> uint16_t(ib32 << 1);
    const uint scale = bitfieldInsert(scale_lo, scale_hi, 4, 2);
    return row_scale * float(int(scale) - 32);
#elif defined(DATA_A_IQ3_KS)
    const bool high_half = bool(ib32 & uint8_t(4));
    const uint8_t group = ib32 & uint8_t(3);
    const uint8_t scale_lo = test_bit_to_get_high4(ib32, 2, iqks_load_u8(block_offset + 2 + group));
    const uint16_t scale_hi = bool(iqks_load_u16(block_offset) & (uint16_t(1) << ib32)) ? uint16_t(0) : uint16_t(-16);
    const int16_t scale = int16_t(uint16_t(scale_lo) | scale_hi);
    return row_scale * float(scale);
#else
    const uint8_t scale = iqks_load_u8(block_offset + ib32);
    return row_scale * float(int(scale & uint8_t(254)) - 127);
#endif
}

#if defined(DATA_A_IQ1_KT) || defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT) || defined(DATA_A_IQ4_KT)
i32vec4 iqkt_unpack16(uint block_byte_offset, uint element) {
    return i32vec4(iqks_value4(block_byte_offset, uint8_t(element)),
                   iqks_value4(block_byte_offset, uint8_t(element + 4u)),
                   iqks_value4(block_byte_offset, uint8_t(element + 8u)),
                   iqks_value4(block_byte_offset, uint8_t(element + 12u)));
}

int8_t iqkt_scale_i8(uint block_byte_offset, uint ib32) {
#if defined(DATA_A_IQ1_KT)
    const uint8_t scale = iqks_load_u8(block_byte_offset + ib32) & uint8_t(0x0F);
    return int8_t(kvalues_iqkt_scale[scale]);
#elif defined(DATA_A_IQ2_KT)
    const uint8_t scale_packed = iqks_load_u8(block_byte_offset + (ib32 & 3u));
    const uint8_t scale = (scale_packed >> (4u * (ib32 >> 2u))) & uint8_t(0x0F);
    return int8_t(kvalues_iqkt_scale[scale]);
#elif defined(DATA_A_IQ3_KT)
    const uint8_t scale_packed = iqks_load_u8(block_byte_offset + (ib32 & 3u));
    return int8_t((scale_packed >> (4u * (ib32 >> 2u))) & uint8_t(0x0F));
#else
    return int8_t(int(iqks_load_u8(block_byte_offset + (ib32 << 2u)) >> 1u) - 64);
#endif
}

void rm_prepare_a_k32x2_iter_shifted(uint block_byte_offset, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t laneid = uint16_t(gl_SubgroupInvocationID);
    const uint16_t row = laneid & uint16_t(3);
    const uint16_t subid = laneid >> uint16_t(2);
    const uint block_base = uint(subid) << 1u;
    const uint block0 = block_base + ((uint(row) & 2u) >> 1u);
    const uint block1 = block_base + (((uint(row) ^ 2u) & 2u) >> 1u);
    const uint half0 = (uint(row) & 1u) << 4u;
    const uint half1 = ((uint(row) ^ 1u) & 1u) << 4u;

    qs_a[0] = iqkt_unpack16(block_byte_offset, (block0 << 5u) + half0);
    qs_a[1] = iqkt_unpack16(block_byte_offset, (block0 << 5u) + half1);
    qs_a[2] = iqkt_unpack16(block_byte_offset, (block1 << 5u) + half0);
    qs_a[3] = iqkt_unpack16(block_byte_offset, (block1 << 5u) + half1);

    const int8_t scale0 = iqkt_scale_i8(block_byte_offset, block_base);
    const int8_t scale1 = iqkt_scale_i8(block_byte_offset, block_base + 1u);
    a_scales = i8vec4(scale0, scale0, scale1, scale1);
}
#define SG_A_SHAPE_32x2
#endif

#if defined(DATA_A_IQ2_KS)
void rm_prepare_a_k16x4_iter_shifted(uint block_byte_offset, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t laneid = uint16_t(gl_SubgroupInvocationID);
    const uint16_t subid = laneid >> uint16_t(2);
    const bool is_high_half = bool(subid & uint16_t(2));
    const uint16_t qs_offset = laneid & uint16_t(0x0c);
    const uint16_t extra = iqks_load_u16(block_byte_offset);
    const uint qs_offset_bytes = block_byte_offset + 6u + 4u * qs_offset;
    const uvec4 qs = uvec4(iqks_load_u32(qs_offset_bytes),
                           iqks_load_u32(qs_offset_bytes + 4u),
                           iqks_load_u32(qs_offset_bytes + 8u),
                           iqks_load_u32(qs_offset_bytes + 12u));
    const uint16_t scale_packed = iqks_load_u16(block_byte_offset + 2u + (is_high_half ? 2u : 0u));
    const uint scale_lo4 = u4x4_to_u8x4(scale_packed);
    const uint e = extra >> (is_high_half ? 12u : 8u);
    const uint neg_scale_hi4bit = (~e) & 0x0fu;
    const uint scale_hi4 = ((neg_scale_hi4bit * 0x00204081u) & 0x01010101u) * 0xf0u;
    a_scales = unpack8(int32_t(scale_lo4 | scale_hi4));
    const uint16_t row = laneid & uint16_t(3);

    [[unroll]] for (uint16_t i = uint16_t(0); i < uint16_t(4); ++i) {
        const uint16_t preshift_i = i ^ row;
        const uint16_t shift = preshift_i << 1;
        const uint16_t extra_shift_idx = preshift_i + (is_high_half ? uint16_t(4) : uint16_t(0));
        const bool is_hi_table = bool((extra >> extra_shift_idx) & uint16_t(1));

        qs_a[i] = i32vec4(unpack_iq2_k((qs.x >> shift) & 0x03030303u, is_hi_table),
                          unpack_iq2_k((qs.y >> shift) & 0x03030303u, is_hi_table),
                          unpack_iq2_k((qs.z >> shift) & 0x03030303u, is_hi_table),
                          unpack_iq2_k((qs.w >> shift) & 0x03030303u, is_hi_table));
    }
}
#define SG_A_SHAPE_16x4
#endif

#if defined(DATA_A_IQ3_KS)
void rm_prepare_a_k16x4_iter_shifted(uint block_byte_offset, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t laneid = uint16_t(gl_SubgroupInvocationID);
    const uint16_t subid = laneid >> 2;
    const bool is_high_16 = bool(subid & uint16_t(1));
    const bool is_high_half = bool(subid & uint16_t(2));
    const uint16_t qs_offset = (is_high_half ? uint16_t(32) : uint16_t(0)) +
                               (is_high_16 ? uint16_t(16) : uint16_t(0));
    const uint16_t qh_offset = is_high_16 ? uint16_t(16) : uint16_t(0);
    const uint qs_offset_bytes = block_byte_offset +  6u + qs_offset;
    const uint qh_offset_bytes = block_byte_offset + 70u + qh_offset;
    const uint ql0 = pack32(u16vec2(iqks_load_u16(qs_offset_bytes      ), iqks_load_u16(qs_offset_bytes +  2u)));
    const uint ql1 = pack32(u16vec2(iqks_load_u16(qs_offset_bytes +  4u), iqks_load_u16(qs_offset_bytes +  6u)));
    const uint ql2 = pack32(u16vec2(iqks_load_u16(qs_offset_bytes +  8u), iqks_load_u16(qs_offset_bytes + 10u)));
    const uint ql3 = pack32(u16vec2(iqks_load_u16(qs_offset_bytes + 12u), iqks_load_u16(qs_offset_bytes + 14u)));
    const uint qh0 = pack32(u16vec2(iqks_load_u16(qh_offset_bytes      ), iqks_load_u16(qh_offset_bytes +  2u)));
    const uint qh1 = pack32(u16vec2(iqks_load_u16(qh_offset_bytes +  4u), iqks_load_u16(qh_offset_bytes +  6u)));
    const uint qh2 = pack32(u16vec2(iqks_load_u16(qh_offset_bytes +  8u), iqks_load_u16(qh_offset_bytes + 10u)));
    const uint qh3 = pack32(u16vec2(iqks_load_u16(qh_offset_bytes + 12u), iqks_load_u16(qh_offset_bytes + 14u)));
    const uint16_t extra = iqks_load_u16(block_byte_offset);
    const uint scale_group_offset = is_high_half ? 4u : 0u;
    const uint scale_packed = iqks_load_u32(block_byte_offset + 2u);
    const uint scale_lo4 = (scale_packed >> (is_high_half ? 4u : 0u)) & 0x0f0f0f0fu;
    const uint e = extra >> (is_high_half ? 4u : 0u);
    const uint neg_scale_hi4bit = (~e) & 0x0fu;
    const uint scale_hi4 = ((neg_scale_hi4bit * 0x00204081u) & 0x01010101u) * 0xf0u;
    a_scales = unpack8(int32_t(scale_lo4 | scale_hi4));
    const uint16_t row = laneid & uint16_t(3);

    [[unroll]] for (uint16_t i = uint16_t(0); i < uint16_t(4); ++i) {
        const uint16_t preshift_i = i ^ row;
        const uint16_t shift_h = preshift_i + uint16_t(scale_group_offset);
        const uint16_t shift_l = preshift_i << 1;
        const bool is_hi_table = bool((extra >> (shift_h + uint16_t(8))) & uint16_t(1));

        qs_a[i] = i32vec4(unpack_iq3_k(ql0 >> shift_l, qh0 >> shift_h, is_hi_table),
                          unpack_iq3_k(ql1 >> shift_l, qh1 >> shift_h, is_hi_table),
                          unpack_iq3_k(ql2 >> shift_l, qh2 >> shift_h, is_hi_table),
                          unpack_iq3_k(ql3 >> shift_l, qh3 >> shift_h, is_hi_table));
    }
}
#define SG_A_SHAPE_16x4
#endif

#if defined(DATA_A_IQ4_KS)
void rm_prepare_a_k32x2_iter_shifted(uint block_byte_offset, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t laneid = uint16_t(gl_SubgroupInvocationID);
    const uint16_t row = laneid & uint16_t(3);
    const uint16_t subid = laneid >> 2;
    const uint16_t ib32 = subid << 1;
    const uint qs_offset = block_byte_offset + 8u + (ib32 << 4);

    const bool swap_qs_halves = bool(row & uint16_t(2));
    const uint qs_offset0 = qs_offset + (swap_qs_halves ? 16u : 0u);
    const uint qs_offset1 = qs_offset + (swap_qs_halves ? 0u : 16u);
    const uvec4 qs0 = uvec4(iqks_load_u32(qs_offset0 +  0u),
                            iqks_load_u32(qs_offset0 +  4u),
                            iqks_load_u32(qs_offset0 +  8u),
                            iqks_load_u32(qs_offset0 + 12u));
    const uvec4 qs1 = uvec4(iqks_load_u32(qs_offset1 +  0u),
                            iqks_load_u32(qs_offset1 +  4u),
                            iqks_load_u32(qs_offset1 +  8u),
                            iqks_load_u32(qs_offset1 + 12u));
    const uint16_t scale_packed = iqks_load_u16(block_byte_offset + ib32);
    const u8vec2 scale_bytes = unpack8(scale_packed).xy;
    const uint16_t scale_lo = uint16_t(scale_bytes.x) * uint16_t(0x0101);
    const uint16_t scale_hi = uint16_t(scale_bytes.y) * uint16_t(0x0101);
    const uint scale_repeated = pack32(u16vec2(scale_lo, scale_hi));
    a_scales = unpack8(int32_t((scale_repeated & 0xfefefefeu) ^ 0x81818181u));

    const uint16_t table_bits = ((scale_lo & uint16_t(1)) * uint16_t(3)) |
                                ((scale_hi & uint16_t(1)) * uint16_t(12));
    const uint16_t shift0 = (row & uint16_t(1)) << 2;
    const uint16_t shift1 = ((row ^ uint16_t(1)) & uint16_t(1)) << 2;
    const bool is_hi_table0 = bool((table_bits >> row) & 1u);
    const bool is_hi_table1 = bool((table_bits >> (row ^ uint16_t(1))) & 1u);
    const bool is_hi_table2 = bool((table_bits >> (row ^ uint16_t(2))) & 1u);
    const bool is_hi_table3 = bool((table_bits >> (row ^ uint16_t(3))) & 1u);

    qs_a[0] = i32vec4(unpack_iq4_k(qs0.x >> shift0, is_hi_table0),
                      unpack_iq4_k(qs0.y >> shift0, is_hi_table0),
                      unpack_iq4_k(qs0.z >> shift0, is_hi_table0),
                      unpack_iq4_k(qs0.w >> shift0, is_hi_table0));
    qs_a[1] = i32vec4(unpack_iq4_k(qs0.x >> shift1, is_hi_table1),
                      unpack_iq4_k(qs0.y >> shift1, is_hi_table1),
                      unpack_iq4_k(qs0.z >> shift1, is_hi_table1),
                      unpack_iq4_k(qs0.w >> shift1, is_hi_table1));
    qs_a[2] = i32vec4(unpack_iq4_k(qs1.x >> shift0, is_hi_table2),
                      unpack_iq4_k(qs1.y >> shift0, is_hi_table2),
                      unpack_iq4_k(qs1.z >> shift0, is_hi_table2),
                      unpack_iq4_k(qs1.w >> shift0, is_hi_table2));
    qs_a[3] = i32vec4(unpack_iq4_k(qs1.x >> shift1, is_hi_table3),
                      unpack_iq4_k(qs1.y >> shift1, is_hi_table3),
                      unpack_iq4_k(qs1.z >> shift1, is_hi_table3),
                      unpack_iq4_k(qs1.w >> shift1, is_hi_table3));
}
#define SG_A_SHAPE_32x2
#endif

#if defined(DATA_A_IQ4_KSS)
void rm_prepare_a_k32x2_iter_shifted(uint block_byte_offset, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t laneid = uint16_t(gl_SubgroupInvocationID);
    const uint16_t row = laneid & uint16_t(3);
    const uint16_t subid = laneid >> 2;
    const uint16_t ib32 = subid << 1;
    const uint qs_offset = block_byte_offset + (ib32 << 4);
    const bool swap_qs_halves = bool(row & uint16_t(2));
    const uint qs_offset0 = qs_offset + (swap_qs_halves ? 16u : 0u);
    const uint qs_offset1 = qs_offset + (swap_qs_halves ? 0u : 16u);

    const uvec4 qs0_raw = uvec4(iqks_load_u32(qs_offset0 +  0u),
                                iqks_load_u32(qs_offset0 +  4u),
                                iqks_load_u32(qs_offset0 +  8u),
                                iqks_load_u32(qs_offset0 + 12u));
    const uvec4 qs1_raw = uvec4(iqks_load_u32(qs_offset1 +  0u),
                                iqks_load_u32(qs_offset1 +  4u),
                                iqks_load_u32(qs_offset1 +  8u),
                                iqks_load_u32(qs_offset1 + 12u));

    const uint scale_bits0 = (qs0_raw.x & 0x00010001u) |
                             ((qs0_raw.y & 0x00010001u) << 2) |
                             ((qs0_raw.z & 0x00010001u) << 4) |
                             ((qs0_raw.w & 0x00010001u) << 6);
    const uint scale_bits1 = (qs1_raw.x & 0x00010001u) |
                             ((qs1_raw.y & 0x00010001u) << 2) |
                             ((qs1_raw.z & 0x00010001u) << 4) |
                             ((qs1_raw.w & 0x00010001u) << 6);
    const uint16_t scale0 = uint16_t(dotPacked4x8EXT(scale_bits0, 0x00020001u));
    const uint16_t scale1 = uint16_t(dotPacked4x8EXT(scale_bits1, 0x00020001u));
    const uint16_t scale0_repeated = scale0 * uint16_t(0x0101);
    const uint16_t scale1_repeated = scale1 * uint16_t(0x0101);
    const uint scale_repeated = pack32(u16vec2(scale0_repeated, scale1_repeated));
    a_scales = unpack8(int32_t((scale_repeated & 0xfefefefeu) ^ 0x81818181u));

    const uvec4 qs0_masked = qs0_raw & 0xfffefffeu;
    const uvec4 qs1_masked = qs1_raw & 0xfffefffeu;
    const uvec4 qs0 = qs0_masked ^ (qs0_masked >> 1u);
    const uvec4 qs1 = qs1_masked ^ (qs1_masked >> 1u);

    const uint16_t shift0 = (row & uint16_t(1)) << 2;
    const uint16_t shift1 = ((row ^ uint16_t(1)) & uint16_t(1)) << 2;
    const bool is_hi_table0 = bool(scale0 & uint16_t(1u));
    const bool is_hi_table1 = bool(scale0 & uint16_t(1u));
    const bool is_hi_table2 = bool(scale1 & uint16_t(1u));
    const bool is_hi_table3 = bool(scale1 & uint16_t(1u));

    qs_a[0] = i32vec4(unpack_iq4_k(qs0.x >> shift0, is_hi_table0),
                      unpack_iq4_k(qs0.y >> shift0, is_hi_table0),
                      unpack_iq4_k(qs0.z >> shift0, is_hi_table0),
                      unpack_iq4_k(qs0.w >> shift0, is_hi_table0));
    qs_a[1] = i32vec4(unpack_iq4_k(qs0.x >> shift1, is_hi_table1),
                      unpack_iq4_k(qs0.y >> shift1, is_hi_table1),
                      unpack_iq4_k(qs0.z >> shift1, is_hi_table1),
                      unpack_iq4_k(qs0.w >> shift1, is_hi_table1));
    qs_a[2] = i32vec4(unpack_iq4_k(qs1.x >> shift0, is_hi_table2),
                      unpack_iq4_k(qs1.y >> shift0, is_hi_table2),
                      unpack_iq4_k(qs1.z >> shift0, is_hi_table2),
                      unpack_iq4_k(qs1.w >> shift0, is_hi_table2));
    qs_a[3] = i32vec4(unpack_iq4_k(qs1.x >> shift1, is_hi_table3),
                      unpack_iq4_k(qs1.y >> shift1, is_hi_table3),
                      unpack_iq4_k(qs1.z >> shift1, is_hi_table3),
                      unpack_iq4_k(qs1.w >> shift1, is_hi_table3));
}
#define SG_A_SHAPE_32x2
#define SG_A_SCALE_SHIFTED
#endif

#if defined(DATA_A_IQ2_KL)
void rm_prepare_a_k32x2_iter_shifted(uint block_byte_offset, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t laneid = uint16_t(gl_SubgroupInvocationID);
    const uint16_t row = laneid & uint16_t(3);
    const uint16_t subid = laneid >> 2u;
    const uint16_t ib32 = subid << 1u;
    const uint qs_offset = block_byte_offset + 6u + (ib32 << 3u);
    const uint qh_offset = block_byte_offset + 70u;

    const uint qs_offset0 = qs_offset + ((row & uint16_t(1)) << 3u);
    const uint qs_offset1 = qs_offset + (((row ^ uint16_t(1)) & uint16_t(1)) << 3u);
    const uint qh_offset0 = qh_offset + ((row & uint16_t(1)) << 3u);
    const uint qh_offset1 = qh_offset + (((row ^ uint16_t(1)) & uint16_t(1)) << 3u);
    const uvec2 qs0 = uvec2(iqks_load_u32(qs_offset0 + 0u),
                            iqks_load_u32(qs_offset0 + 4u));
    const uvec2 qs1 = uvec2(iqks_load_u32(qs_offset1 + 0u),
                            iqks_load_u32(qs_offset1 + 4u));
    const uvec2 qh0 = uvec2(iqks_load_u32(qh_offset0 + 0u),
                            iqks_load_u32(qh_offset0 + 4u));
    const uvec2 qh1 = uvec2(iqks_load_u32(qh_offset1 + 0u),
                            iqks_load_u32(qh_offset1 + 4u));

    const uint16_t block0 = ib32 + ((row & uint16_t(2)) >> 1u);
    const uint16_t block1 = ib32 + (((row ^ uint16_t(2)) & uint16_t(2)) >> 1u);
    const i32vec2 values00 = unpack_iq2_kl(qs0.x, qh0.x, block0);
    const i32vec2 values01 = unpack_iq2_kl(qs0.y, qh0.y, block0);
    const i32vec2 values10 = unpack_iq2_kl(qs1.x, qh1.x, block0);
    const i32vec2 values11 = unpack_iq2_kl(qs1.y, qh1.y, block0);
    const i32vec2 values20 = unpack_iq2_kl(qs0.x, qh0.x, block1);
    const i32vec2 values21 = unpack_iq2_kl(qs0.y, qh0.y, block1);
    const i32vec2 values30 = unpack_iq2_kl(qs1.x, qh1.x, block1);
    const i32vec2 values31 = unpack_iq2_kl(qs1.y, qh1.y, block1);

    qs_a[0] = i32vec4(values00.x, values00.y, values01.x, values01.y);
    qs_a[1] = i32vec4(values10.x, values10.y, values11.x, values11.y);
    qs_a[2] = i32vec4(values20.x, values20.y, values21.x, values21.y);
    qs_a[3] = i32vec4(values30.x, values30.y, values31.x, values31.y);

    const uint scale_offset = block_byte_offset + 2u + uint(ib32 & uint16_t(3));
    const uint16_t scale_l = iqks_load_u16(scale_offset);
    const uint16_t scale_h = iqks_load_u16(block_byte_offset) >> (ib32 << 1u);
    const uint16_t scale_shift = ib32 & uint16_t(4);
    const uint16_t scale_nibbles = (scale_l >> scale_shift) & uint16_t(0x0f0f);
    const uint scale_lo = u4x4_to_u8x4(scale_nibbles * uint16_t(0x11));
    const uint16_t scale_h_repeated = ((scale_h & uint16_t(3)) * uint16_t(5)) |
                                      (((scale_h >> 2u) & uint16_t(3)) * uint16_t(0x50));
    const uint scale_hi = u2x4_to_u8x4_sub32(scale_h_repeated);
    const i8vec4 scale_values = unpack8(int32_t(scale_lo | scale_hi));
    a_scales = bool(row & uint16_t(2)) ? scale_values.zwxy : scale_values;
}
#define SG_A_SHAPE_32x2
#define SG_A_SCALE_SHIFTED
#endif

#if defined(DATA_A_IQ5_KS)
void rm_prepare_a_k32x2_iter_shifted(uint block_byte_offset, out i32vec4 qs_a[4], out i8vec4 a_scales) {
    const uint16_t laneid = uint16_t(gl_SubgroupInvocationID);
    const uint16_t row = laneid & uint16_t(3);
    const uint16_t subid = laneid >> uint16_t(2);
    const uint16_t ib32 = subid << uint16_t(1);
    const uint qs_offset = block_byte_offset + 8u + (ib32 << 4u);
    const uint qh_offset = block_byte_offset + 136u;

    const uint qs_offset0 = qs_offset + ((row & uint16_t(1)) << 4u);
    const uint qs_offset1 = qs_offset + (((row ^ uint16_t(1)) & uint16_t(1)) << 4u);
    const uint qh_offset0 = qh_offset + ((row & uint16_t(1)) << 4u);
    const uint qh_offset1 = qh_offset + (((row ^ uint16_t(1)) & uint16_t(1)) << 4u);
    const uvec4 qs0 = uvec4(iqks_load_u32(qs_offset0 +  0u),
                            iqks_load_u32(qs_offset0 +  4u),
                            iqks_load_u32(qs_offset0 +  8u),
                            iqks_load_u32(qs_offset0 + 12u));
    const uvec4 qs1 = uvec4(iqks_load_u32(qs_offset1 +  0u),
                            iqks_load_u32(qs_offset1 +  4u),
                            iqks_load_u32(qs_offset1 +  8u),
                            iqks_load_u32(qs_offset1 + 12u));
    const uvec4 qh0 = uvec4(iqks_load_u32(qh_offset0 +  0u),
                            iqks_load_u32(qh_offset0 +  4u),
                            iqks_load_u32(qh_offset0 +  8u),
                            iqks_load_u32(qh_offset0 + 12u));
    const uvec4 qh1 = uvec4(iqks_load_u32(qh_offset1 +  0u),
                            iqks_load_u32(qh_offset1 +  4u),
                            iqks_load_u32(qh_offset1 +  8u),
                            iqks_load_u32(qh_offset1 + 12u));

    const uint16_t block0 = ib32 + ((row & uint16_t(2)) >> 1u);
    const uint16_t block1 = ib32 + (((row ^ uint16_t(2)) & uint16_t(2)) >> 1u);
    const uint16_t scale_packed = iqks_load_u16(block_byte_offset + ib32);
    const u8vec2 scale_bytes = unpack8(scale_packed).xy;
    const uint16_t scale_lo = uint16_t(scale_bytes.x) * uint16_t(0x0101);
    const uint16_t scale_hi = uint16_t(scale_bytes.y) * uint16_t(0x0101);
    const uint scale_repeated = pack32(u16vec2(scale_lo, scale_hi));
    a_scales = unpack8(int32_t((scale_repeated & 0xfefefefeu) ^ 0x81818181u));

    const uint16_t scale_table_bits = (scale_packed | (scale_packed >> 7u)) & uint16_t(3);
    const uint table_offset0 = ((scale_table_bits >> (block0 - ib32)) & uint16_t(1)) << 5u;
    const uint table_offset1 = table_offset0;
    const uint table_offset2 = ((scale_table_bits >> (block1 - ib32)) & uint16_t(1)) << 5u;
    const uint table_offset3 = table_offset2;

    qs_a[0] = i32vec4(unpack_iq5_k(qs0.x, qh0.x, block0, table_offset0),
                      unpack_iq5_k(qs0.y, qh0.y, block0, table_offset0),
                      unpack_iq5_k(qs0.z, qh0.z, block0, table_offset0),
                      unpack_iq5_k(qs0.w, qh0.w, block0, table_offset0));
    qs_a[1] = i32vec4(unpack_iq5_k(qs1.x, qh1.x, block0, table_offset1),
                      unpack_iq5_k(qs1.y, qh1.y, block0, table_offset1),
                      unpack_iq5_k(qs1.z, qh1.z, block0, table_offset1),
                      unpack_iq5_k(qs1.w, qh1.w, block0, table_offset1));
    qs_a[2] = i32vec4(unpack_iq5_k(qs0.x, qh0.x, block1, table_offset2),
                      unpack_iq5_k(qs0.y, qh0.y, block1, table_offset2),
                      unpack_iq5_k(qs0.z, qh0.z, block1, table_offset2),
                      unpack_iq5_k(qs0.w, qh0.w, block1, table_offset2));
    qs_a[3] = i32vec4(unpack_iq5_k(qs1.x, qh1.x, block1, table_offset3),
                      unpack_iq5_k(qs1.y, qh1.y, block1, table_offset3),
                      unpack_iq5_k(qs1.z, qh1.z, block1, table_offset3),
                      unpack_iq5_k(qs1.w, qh1.w, block1, table_offset3));
}
#define SG_A_SHAPE_32x2
#endif

#endif

#if defined(SG_IQK)
#if defined(DATA_A_IQK_ROW)
#define ROW_META_TYPE float

ROW_META_TYPE rm_prepare_row_meta(uint row_begin_offset) {
    return iqks_row_scale(row_begin_offset);
}
#endif

#if defined(DATA_A_IQ2_KS)
#define SG_IQK_LANES_PER_32 (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_HALF (QUANT_K_Q8_1 / 4)

void rm_prepare_a_sg(uint block_byte_offset, ROW_META_TYPE row_meta, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const uint16_t extra = iqks_load_u16(block_byte_offset);
    const uint qs_u32 = iqks_load_u32(block_byte_offset + 6u + 4u * (uint(lane) % 16u));

    const uint16_t q_offset = uint16_t((lane * SG_QS_PER_ITER) % SG_IQK_QS_PER_HALF);
    const uint16_t qs_base = lane >= (SUBGROUP_SIZE / 2u) ? uint16_t(SG_IQK_QS_PER_HALF) : uint16_t(0);
    const uint16_t shift_l = uint16_t(2u * (ib32 % 4u));
    const bool is_hi_table = bool((extra >> ib32) & uint16_t(1));

    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q32 = q_offset + q;
        const uint q2 = subgroupShuffle(qs_u32, qs_base + q32);
        qs_a[q] = unpack_iq2_k((q2 >> shift_l) & 0x03030303u, is_hi_table);
    }

    const uint8_t scale_l = iqks_load_u8(block_byte_offset + 2u + (uint(ib32) >> 1u));
    const bool scale_high_half = bool(ib32 & uint16_t(1));
    const uint8_t scale = (scale_l >> (scale_high_half ? 4u : 0u)) & uint8_t(0x0F);
    const bool scale_high = bool((extra >> (uint(ib32) + 8u)) & uint16_t(1));
    a_scale = row_meta * float(int32_t(scale) + (scale_high ? 0 : -16));
}
#elif defined(DATA_A_IQ3_KS)
#define SG_IQK_LANES_PER_32 (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_HALF (QUANT_K_Q8_1 / 4)

void rm_prepare_a_sg(uint block_byte_offset, ROW_META_TYPE row_meta, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const uint16_t extra = iqks_load_u16(block_byte_offset);
    const uint qs_offset = 2u * (uint(lane) % 16u);
    const uint qs_u32 = pack32(u16vec2(iqks_load_u16(block_byte_offset + 6u + 2u * qs_offset),
                                       iqks_load_u16(block_byte_offset + 8u + 2u * qs_offset)));
    const uint qh_offset = 2u * (uint(lane) % 8u);
    const uint qh_u32 = pack32(u16vec2(iqks_load_u16(block_byte_offset + 70u + 2u * qh_offset),
                                       iqks_load_u16(block_byte_offset + 72u + 2u * qh_offset)));

    const uint16_t q_offset = uint16_t((lane * SG_QS_PER_ITER) % SG_IQK_QS_PER_HALF);
    const uint16_t qs_base = lane >= (SUBGROUP_SIZE / 2u) ? uint16_t(SG_IQK_QS_PER_HALF) : uint16_t(0);
    const uint16_t shift_h = ib32;
    const uint16_t shift_l = uint16_t(2u * (ib32 % 4u));
    const bool is_hi_table = bool((uint(extra) >> (uint(ib32) + 8u)) & 1u);

    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q32 = q_offset + q;
        const uint ql = subgroupShuffle(qs_u32, qs_base + q32);
        const uint qh = subgroupShuffle(qh_u32, q32);
        qs_a[q] = unpack_iq3_k(ql, qh, shift_h, is_hi_table);
    }

    const uint8_t scale_l = iqks_load_u8(block_byte_offset + 2u + (uint(ib32) & 3u));
    const uint scale = (uint(scale_l) >> (ib32 >= 4u ? 4u : 0u)) & 0x0Fu;
    const uint scale_high = (uint(extra) >> uint(ib32)) & 1u;
    a_scale = row_meta * float(int32_t(scale | (scale_high << 4u)) - 16);
}
#elif defined(DATA_A_IQ4_KS)
#define SG_IQK_LANES_PER_32 (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_32 (QUANT_K_Q8_1 / 8)

void rm_prepare_a_sg(uint block_byte_offset, ROW_META_TYPE row_meta, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const bool scale_high_half = bool(lane & uint16_t(SG_IQK_LANES_PER_32 / 2u));
    const uint16_t q_offset = uint16_t((uint(lane) * SG_QS_PER_ITER) % SG_IQK_QS_PER_32);
    const uint8_t scale = iqks_load_u8(block_byte_offset + uint(ib32));

#if SUBGROUP_SIZE == 16
    const uvec2 qs_u32 = uvec2(iqks_load_u32(block_byte_offset + 8u + uint(lane) * 8u),
                               iqks_load_u32(block_byte_offset + 12u + uint(lane) * 8u));
#else
    const uint qs_u32 = iqks_load_u32(block_byte_offset + 8u + 4u * (uint(lane) & 31u));
#endif

    const bool is_hi_table = bool(uint(scale) & 1u);
    const uint shift = scale_high_half ? 4u : 0u;

#if SUBGROUP_SIZE == 16
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; q += 2u) {
        const uint q_idx = 4u * uint(ib32) + uint(q_offset) + q;
        const uvec2 qs_pair = subgroupShuffle(qs_u32, q_idx >> 1u);
        qs_a[q] = unpack_iq4_k(qs_pair.x, shift, is_hi_table);
        qs_a[q + 1u] = unpack_iq4_k(qs_pair.y, shift, is_hi_table);
    }
#else
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q_idx = 4u * uint(ib32) + uint(q_offset) + q;
        const uint q4 = subgroupShuffle(qs_u32, q_idx);
        qs_a[q] = unpack_iq4_k(q4, shift, is_hi_table);
    }
#endif

    a_scale = row_meta * float(int32_t(uint(scale) & 254u) - 127);
}
#elif defined(DATA_A_IQ5_KS)
#define SG_IQK_LANES_PER_32 (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_32 (QUANT_K_Q8_1 / 4)

void rm_prepare_a_sg(uint block_byte_offset, ROW_META_TYPE row_meta, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const uint16_t q_offset = uint16_t((uint(lane) * SG_QS_PER_ITER) % SG_IQK_QS_PER_32);
    const uint8_t scale = iqks_load_u8(block_byte_offset + uint(ib32));

#if SUBGROUP_SIZE == 16
    const uvec2 qs_u32 = uvec2(iqks_load_u32(block_byte_offset + 8u + uint(lane) * 8u),
                               iqks_load_u32(block_byte_offset + 12u + uint(lane) * 8u));
#else
    const uint qs_u32 = iqks_load_u32(block_byte_offset + 8u + 4u * (uint(lane) & 31u));
#endif
    const uint qh_u32 = iqks_load_u32(block_byte_offset + 136u + 4u * (uint(lane) & 7u));

    const uint table_offset = (uint(scale) & 1u) << 5u;

#if SUBGROUP_SIZE == 16
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; q += 2u) {
        const uint q_idx = 8u * (uint(ib32) / 2u) + uint(q_offset) + q;
        const uvec2 qs_pair = subgroupShuffle(qs_u32, q_idx >> 1u);
        const uint qh0 = subgroupShuffle(qh_u32, uint(q_offset) + q);
        const uint qh1 = subgroupShuffle(qh_u32, uint(q_offset) + q + 1u);
        qs_a[q] = unpack_iq5_k(qs_pair.x, qh0, uint(ib32), table_offset);
        qs_a[q + 1u] = unpack_iq5_k(qs_pair.y, qh1, uint(ib32), table_offset);
    }
#else
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q_idx = 8u * (uint(ib32) / 2u) + uint(q_offset) + q;
        const uint qs = subgroupShuffle(qs_u32, q_idx);
        const uint qh = subgroupShuffle(qh_u32, uint(q_offset) + q);
        qs_a[q] = unpack_iq5_k(qs, qh, uint(ib32), table_offset);
    }
#endif

    a_scale = row_meta * float(int32_t(uint(scale) & 254u) - 127);
}
#endif
#endif