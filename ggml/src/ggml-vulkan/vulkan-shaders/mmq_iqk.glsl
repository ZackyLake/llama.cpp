
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