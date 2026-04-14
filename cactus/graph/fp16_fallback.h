#ifndef GRAPH_FP16_FALLBACK_H
#define GRAPH_FP16_FALLBACK_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace Fp16Fallback {

inline float fp16_bits_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;

    uint32_t out_bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            out_bits = sign;
        } else {
            int32_t e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03FFu;
            out_bits = sign |
                       static_cast<uint32_t>((e + 127) << 23) |
                       (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        out_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        out_bits = sign |
                   ((exp + 112u) << 23) |
                   (mant << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &out_bits, sizeof(out));
    return out;
}

inline uint16_t float_to_fp16_bits(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));

    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exp = (bits >> 23) & 0xFFu;
    const uint32_t mant = bits & 0x7FFFFFu;

    if (exp == 0xFFu) {
        if (mant != 0) {
            uint16_t nan_mant = static_cast<uint16_t>(mant >> 13);
            if (nan_mant == 0) nan_mant = 1;
            return static_cast<uint16_t>(sign | 0x7C00u | nan_mant);
        }
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    const int32_t exp_unbiased = static_cast<int32_t>(exp) - 127;
    int32_t half_exp = exp_unbiased + 15;

    if (half_exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    if (half_exp <= 0) {
        if (half_exp < -10) {
            return sign;
        }

        uint32_t mantissa = mant | 0x800000u;
        const int32_t shift = 14 - half_exp;
        uint16_t half_mant = static_cast<uint16_t>(mantissa >> shift);
        const uint32_t round_bit = 1u << (shift - 1);
        const uint32_t round_mask = round_bit - 1u;
        if ((mantissa & round_bit) != 0 &&
            (((mantissa & round_mask) != 0) || ((half_mant & 1u) != 0))) {
            ++half_mant;
        }
        return static_cast<uint16_t>(sign | half_mant);
    }

    uint16_t half = static_cast<uint16_t>(sign |
                                          (static_cast<uint16_t>(half_exp) << 10) |
                                          static_cast<uint16_t>(mant >> 13));
    if ((mant & 0x1000u) != 0) {
        ++half;
    }
    return half;
}

inline float load_fp16(const __fp16* src) {
    uint16_t bits = 0;
    std::memcpy(&bits, src, sizeof(bits));
    return fp16_bits_to_float(bits);
}

inline void store_fp16(__fp16* dst, float value) {
    const uint16_t bits = float_to_fp16_bits(value);
    std::memcpy(dst, &bits, sizeof(bits));
}

inline int8_t sign_extend_int4(uint8_t v) {
    v &= 0x0Fu;
    return static_cast<int8_t>((v & 0x08u) ? static_cast<int8_t>(v) - 16 : static_cast<int8_t>(v));
}

inline int8_t load_int8_interleaved(const int8_t* data, size_t K, size_t n, size_t k) {
    const size_t n_block = n / 4;
    const size_t lane = n % 4;
    const size_t idx = ((n_block * K + k) * 4) + lane;
    return data[idx];
}

inline int8_t load_int4_interleaved(const int8_t* packed, size_t K, size_t group_size, size_t n, size_t k) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(packed);
    const size_t n_block = n / 4;
    const size_t lane = n % 4;
    const size_t group_idx = k / group_size;
    const size_t in_group = k % group_size;
    const size_t chunk16 = in_group / 16;
    const size_t pos = in_group % 16;

    const size_t group_base = (n_block * K + group_idx * group_size) * 2;
    const size_t lane_block = (lane >= 2) ? 16 : 0;
    const size_t byte_idx = group_base + chunk16 * 32 + lane_block + pos;

    const uint8_t raw = bytes[byte_idx];
    const uint8_t nibble = (lane % 2 == 0) ? (raw & 0x0Fu) : (raw >> 4);
    return sign_extend_int4(nibble);
}

inline float load_group_scale(const __fp16* scales, size_t num_groups, size_t n, size_t group_idx) {
    const size_t n_block = n / 4;
    const size_t lane = n % 4;
    const size_t s_idx = (n_block * num_groups + group_idx) * 4 + lane;
    return load_fp16(scales + s_idx);
}

inline float quantize_row_fp16_to_int8(const __fp16* src, int8_t* dst, size_t K) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < K; ++i) {
        max_abs = std::max(max_abs, std::fabs(load_fp16(src + i)));
    }
    float scale = max_abs / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    const float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < K; ++i) {
        const float q = load_fp16(src + i) * inv_scale;
        const int32_t qi = static_cast<int32_t>(std::nearbyint(q));
        dst[i] = static_cast<int8_t>(std::max(-128, std::min(127, qi)));
    }
    return scale;
}

inline float load_value(const void* base, Precision p, size_t i) {
    if (p == Precision::FP32) {
        return static_cast<const float*>(base)[i];
    }
    if (p == Precision::FP16) {
        return load_fp16(static_cast<const __fp16*>(base) + i);
    }
    return static_cast<float>(static_cast<const int8_t*>(base)[i]);
}

} // namespace Fp16Fallback

#endif // GRAPH_FP16_FALLBACK_H
