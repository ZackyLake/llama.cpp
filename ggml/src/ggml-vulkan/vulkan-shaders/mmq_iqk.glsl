
#if defined(DATA_A_IQ2_K) || defined(DATA_A_IQ2_KS)

#if 1

shared uint32_t iq2k_table[512];

void init_iq2k_table() {
    const uint16_t workgroup_size = uint16_t(gl_WorkGroupSize.x); // * gl_WorkGroupSize.y * gl_WorkGroupSize.z);
    // assume only gl_WorkGroupSize.x and 256 being multiplies on it.
    for (uint16_t index_ = uint16_t(gl_LocalInvocationIndex); index_ < uint16_t(256u); index_ += workgroup_size) {
        const uint8_t index = uint8_t(index_);

        const uint index_lo = pack32(u8vec4(index_ >> 0u, index_ >> 2u, index_ >> 4u, index_ >> 6u)) & 0x03030303u;
        const uint index_hi = index_lo + 0x04040404u;
        const u8vec4 ql = unpack8(index_lo);
        const u8vec4 qh = unpack8(index_hi);
        iq2k_table[index_] = pack32(i8vec4(
            kvalues_iq2_k[ql.x], kvalues_iq2_k[ql.y], kvalues_iq2_k[ql.z], kvalues_iq2_k[ql.w]));
        iq2k_table[256u + index_] = pack32(i8vec4(
            kvalues_iq2_k[qh.x], kvalues_iq2_k[qh.y], kvalues_iq2_k[qh.z], kvalues_iq2_k[qh.w]));
        //iq2k_table[index_] = pack32(i8vec4(
        //    kvalues_iq2_k_const[ql.x], kvalues_iq2_k_const[ql.y], kvalues_iq2_k_const[ql.z], kvalues_iq2_k_const[ql.w]));
        //iq2k_table[256u + index_] = pack32(i8vec4(
        //    kvalues_iq2_k_const[qh.x], kvalues_iq2_k_const[qh.y], kvalues_iq2_k_const[qh.z], kvalues_iq2_k_const[qh.w]));
    }
    barrier();
}

int32_t unpack_iq2_k(uint32_t values, bool is_hi_table) {
    const uint table_offset = is_hi_table ? 256u : 0u;
    const uint index = table_offset + dotPacked4x8EXT(values, 0x40100401u);
    return int32_t(iq2k_table[index]);
}

#elif 0

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_shader_subgroup_extended_types_int8 : require

int32_t unpack_iq2_k(uint32_t values, bool is_hi_table) {
    if (is_hi_table) values += 0x04040404u;
    const u8vec4 indexes = unpack8(values);
    const int8_t lut = gl_SubgroupInvocationID < 8
        ? kvalues_iq2_k_const[gl_SubgroupInvocationID]
        : int8_t(0);
    return pack32(i8vec4(subgroupShuffle(lut, indexes.x),
                         subgroupShuffle(lut, indexes.y),
                         subgroupShuffle(lut, indexes.z),
                         subgroupShuffle(lut, indexes.w)));
}

#elif 0

int32_t unpack_iq2_k(uint32_t values, bool is_hi_table) {
    if (is_hi_table) values += 0x04040404u;
    const u8vec4 indexes = unpack8(values);
    return pack32(i8vec4(kvalues_iq2_k[indexes.x],
                         kvalues_iq2_k[indexes.y],
                         kvalues_iq2_k[indexes.z],
                         kvalues_iq2_k[indexes.w]));
}

#endif

#endif

#if defined(DATA_A_IQ3_K) || defined(DATA_A_IQ3_KS)

shared uint16_t iq3k_table[128];

void init_iq3k_table() {
    for (uint index = gl_LocalInvocationIndex; index < 64u; index += gl_WorkGroupSize.x) {
        const i8vec2 values = i8vec2(kvalues_iq3_k[index & 7u], kvalues_iq3_k[index >> 3u]);
        iq3k_table[index] = uint16_t(pack16(values));
        iq3k_table[64u + index] = uint16_t(pack16(values + i8vec2(4)));
    }
    barrier();
}

int32_t unpack_iq3_k(uint32_t ql, uint32_t qh, uint8_t shift_h, bool is_hi_table) {
    const uint8_t shift_l = (shift_h & uint8_t(3)) << 1;
    const uint indexes = ((ql >> shift_l) & 0x03030303u) |
                         (((qh >> shift_h) & 0x01010101u) << 2u);
    const uint table_offset = is_hi_table ? 64u : 0u;
    const uint index0 = table_offset + dotPacked4x8EXT(indexes & 0xffffu, 0x00000801u); // WA for intel bug
    const uint index1 = table_offset + dotPacked4x8EXT(indexes, 0x08010000u);
    return int32_t(pack32(u16vec2(iq3k_table[index0], iq3k_table[index1])));
}

#endif

#if defined(DATA_A_IQK_ROW)

uint8_t iqks_load_u8(uint offset) {
    return data_a[offset];
}

uint16_t iqks_load_u16(uint offset) {
#if defined(A_TYPE_PACKED16)
    return data_a_packed16[offset / 2];
#else
    return pack8(u8vec2(iqks_load_u8(offset), iqks_load_u8(offset + 1)));
#endif
}

uint32_t iqks_load_u32(uint offset) {
#if defined(A_TYPE_PACKED32)
    return data_a_packed32[offset / 4];
#else
    return pack32(u16vec2(iqks_load_u16(offset), iqks_load_u16(offset + 2)));
#endif
}

float iqks_row_scale(uint row_offset) {
#if defined(DATA_A_IQ2_KS) || defined(DATA_A_IQ2_KL) || defined(DATA_A_IQ3_KS)
    return unpackHalf2x16(iqks_load_u16(row_offset)).x;
#else
    return uintBitsToFloat(iqks_load_u32(row_offset));
#endif
}

#if defined(DATA_A_IQ1_KT) || defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT) || defined(DATA_A_IQ4_KT)
int8_t iqkt_next(inout uint state) {
    state *= 0xCBAC1FEDu;
    const uint value = state & 0x3F3F3F3Fu;
    return int8_t(dotPacked4x8EXT(int32_t(value), int32_t(0x01010101u)) - 126);
}
#endif

int32_t iqks_value4(uint block_offset, uint element) {
    const uint pos = element % 32;

#if defined(DATA_A_IQ1_KT) || defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT) || defined(DATA_A_IQ4_KT)
    const uint group8 = element / 8;
    const uint pos8 = element % 8;
    const uint ib32 = element / 32;
    uint state;

#if defined(DATA_A_IQ1_KT)
    const uint8_t sh = iqks_load_u8(block_offset + ib32);
    const uint8_t x = iqks_load_u8(block_offset + 8 + group8);
    const uint8_t y = (iqks_load_u8(block_offset + 40 + group8 % 16) >> (4 * (group8 / 16))) & uint8_t(0x0F);
    const bool z = bool(sh & (uint8_t(1) << (4 + group8 % 4)));
    state = pack32(u8vec4(x, y | (z ? uint8_t(2) : uint8_t(1)) << 4, uint8_t(0), uint8_t(0)));
#elif defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT)
    state = uint(iqks_load_u16(block_offset + 4 + 2 * group8)) + 4096;
#else
    const uint header = iqks_load_u32(block_offset + 4 * ib32);
    const uint seed_index = 2 * group8 + pos8 / 4;
    const uint8_t x = iqks_load_u8(block_offset + 32 + seed_index);
    const uint8_t y = (iqks_load_u8(block_offset + 96 + seed_index % 32) >> (4 * (seed_index / 32))) & uint8_t(0x0F);
    const uint8_t z = uint8_t((header >> (8 + 3 * (seed_index % 8))) & 7);
    const bool w = bool(header & 1u);
    const uint8_t high = z + uint8_t(1) + (w ? uint8_t(8) : uint8_t(0));
    state = pack32(u8vec4(x, y | ((high & uint8_t(0x0F)) << 4), high >> 4, uint8_t(0)));
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
    const u8vec4 indexes = unpack8((values >> (4 * (pos / 16))) & 0x0F0F0F0Fu);
    const uint table_offset = bool(scale & uint8_t(1)) ? 16u : 0u;
    return pack32(i8vec4(kvalues_iq4_k[indexes.x + table_offset],
                         kvalues_iq4_k[indexes.y + table_offset],
                         kvalues_iq4_k[indexes.z + table_offset],
                         kvalues_iq4_k[indexes.w + table_offset]));
#elif defined(DATA_A_IQ2_KS)
    const bool high_half = bool(element & 128u);
    const uint group = (element % 128) / 32;
    const uint16_t extra = iqks_load_u16(block_offset) >> (high_half ? 4u : 0u);
    const uint values = iqks_load_u32(block_offset + 6 + (high_half ? 32u : 0u) + pos);
    const bool is_hi_table = bool(extra & (uint16_t(1) << group));
    return unpack_iq2_k((values >> (2 * group)) & 0x03030303u, is_hi_table);
#elif defined(DATA_A_IQ2_KL)
    const uint ib64 = element / 64;
    const uint pos64 = element % 64;
    const uint pos32 = pos64 % 32;
    const bool high_half = bool(pos64 & 32u);
    const bool odd = bool(pos32 & 1u);
    const uint16_t packed = iqks_load_u16(block_offset + 6 + 16 * ib64 + pos32 / 2);
    const uint16_t high = iqks_load_u16(block_offset + 70 + pos32 / 2);
    const uint shift = high_half ? 4u : 0u;
    const uint high_shift = 2 * ib64 + (high_half ? 1u : 0u);
    const uint8_t index0 = uint8_t(((packed >> shift) & uint16_t(0x0F)) | (((high >> high_shift) & uint16_t(1)) << 4));
    const uint8_t index1 = uint8_t((((packed >> 8) >> shift) & uint16_t(0x0F)) | ((((high >> 8) >> high_shift) & uint16_t(1)) << 4));
    const uint lane0 = odd ? 1u : 0u;
    const uint lane1 = odd ? 0u : 1u;
    return pack32(i8vec4(kvalues_iq2_kl[2 * index0 + lane0],
                         kvalues_iq2_kl[2 * index0 + lane1],
                         kvalues_iq2_kl[2 * index1 + lane0],
                         kvalues_iq2_kl[2 * index1 + lane1]));
#elif defined(DATA_A_IQ3_KS)
    const uint8_t shift_h = uint8_t(element) >> 5;
    const uint8_t group = shift_h & uint8_t(3);
    const bool high_half = bool(shift_h & uint8_t(4));
    const uint16_t extra = iqks_load_u16(block_offset) >> (high_half ? 4u : 0u);
    const uint low = iqks_load_u32(block_offset + 6 + (high_half ? 32u : 0u) + pos);
    const uint high = iqks_load_u32(block_offset + 70 + pos);
    const bool is_hi_table = bool(extra & (uint16_t(1) << (8 + group)));
    return unpack_iq3_k(low, high, shift_h, is_hi_table);
#elif defined(DATA_A_IQ4_KS)
    const uint ib32 = element / 32;
    const uint8_t scale = iqks_load_u8(block_offset + ib32);
    const uint values = iqks_load_u32(block_offset + 8 + 16 * ib32 + pos % 16);
    const u8vec4 indexes = unpack8((values >> (4 * (pos / 16))) & 0x0F0F0F0Fu);
    const uint table_offset = bool(scale & uint8_t(1)) ? 16u : 0u;
    return pack32(i8vec4(kvalues_iq4_k[indexes.x + table_offset],
                         kvalues_iq4_k[indexes.y + table_offset],
                         kvalues_iq4_k[indexes.z + table_offset],
                         kvalues_iq4_k[indexes.w + table_offset]));
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

float iqks_d_scale(uint row_offset, uint block_offset, uint ib32) {
#if defined(DATA_A_IQ1_KT)
    const uint8_t scale = iqks_load_u8(block_offset + ib32) & uint8_t(0x0F);
    return iqks_row_scale(row_offset) * float(kvalues_iqkt_scale[scale]);
#elif defined(DATA_A_IQ2_KT)
    const uint8_t scale = (iqks_load_u8(block_offset + ib32 % 4) >> (4 * (ib32 / 4))) & uint8_t(0x0F);
    return iqks_row_scale(row_offset) * float(kvalues_iqkt_scale[scale]);
#elif defined(DATA_A_IQ3_KT)
    const uint8_t scale = (iqks_load_u8(block_offset + ib32 % 4) >> (4 * (ib32 / 4))) & uint8_t(0x0F);
    return iqks_row_scale(row_offset) * float(scale);
#elif defined(DATA_A_IQ4_KT)
    return iqks_row_scale(row_offset) * float(int((iqks_load_u32(block_offset + 4 * ib32) & 0xFF) >> 1) - 64);
#elif defined(DATA_A_IQ4_KSS)
    uint scale_bits = 0;
    [[unroll]] for (uint j = 0; j < 4; ++j) {
        scale_bits |= (iqks_load_u32(block_offset + 4 * (4 * ib32 + j)) & 0x00010001) << (2 * j);
    }
    const uint8_t scale = uint8_t(scale_bits | (scale_bits >> 15));
    return iqks_row_scale(row_offset) * float(int(scale & uint8_t(254)) - 127);
#elif defined(DATA_A_IQ2_KS)
    const bool high_half = bool(ib32 & 4u);
    const uint group = ib32 % 4;
    const uint16_t extra = iqks_load_u16(block_offset) >> (high_half ? 4u : 0u);
    const uint8_t packed_scale = iqks_load_u8(block_offset + 2 + (high_half ? 2u : 0u) + group / 2);
    const uint8_t scale = ((packed_scale >> (4 * (group % 2))) & uint8_t(0x0F)) |
                          (bool(extra & (uint16_t(1) << (8 + group))) ? uint8_t(16) : uint8_t(0));
    return iqks_row_scale(row_offset) * float(int(scale) - 16);
#elif defined(DATA_A_IQ2_KL)
    const uint8_t scale_low = (iqks_load_u8(block_offset + 2 + ib32 % 4) >> (4 * (ib32 / 4))) & uint8_t(0x0F);
    const uint8_t scale_high = uint8_t((iqks_load_u16(block_offset) >> (2 * ib32)) & uint16_t(3));
    return iqks_row_scale(row_offset) * float(int(scale_low | (scale_high << 4)) - 32);
#elif defined(DATA_A_IQ3_KS)
    const bool high_half = bool(ib32 & 4u);
    const uint group = ib32 % 4;
    const uint16_t extra = iqks_load_u16(block_offset) >> (high_half ? 4u : 0u);
    const uint8_t scale = ((iqks_load_u8(block_offset + 2 + group) >> (high_half ? 4u : 0u)) & uint8_t(0x0F)) |
                          (bool(extra & (uint16_t(1) << group)) ? uint8_t(16) : uint8_t(0));
    return iqks_row_scale(row_offset) * float(int(scale) - 16);
#else
    const uint8_t scale = iqks_load_u8(block_offset + ib32);
    return iqks_row_scale(row_offset) * float(int(scale & uint8_t(254)) - 127);
#endif
}

#endif