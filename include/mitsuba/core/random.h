/*
 * Tiny self-contained version of the PCG Random Number Generation for C++ put
 * together from pieces of the much larger C/C++ codebase with vectorization
 * using Enoki.
 *
 * Wenzel Jakob, February 2017
 *
 * The PCG random number generator was developed by Melissa O'Neill
 * <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

#pragma once

#include <mitsuba/core/simd.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/traits.h>
#include <mitsuba/core/fwd.h>
#include <enoki/random.h>

NAMESPACE_BEGIN(enoki)
/// Prints the canonical representation of a PCG32 object.
template <typename Value>
std::ostream& operator<<(std::ostream &os, const enoki::PCG32<Value> &p) {
    os << "PCG32[" << std::endl
       << "  state = 0x" << std::hex << p.state << "," << std::endl
       << "  inc = 0x" << std::hex << p.inc << std::endl
       << "]";
    return os;
}
NAMESPACE_END(enoki)

NAMESPACE_BEGIN(mitsuba)

template <typename UInt32> using PCG32 = std::conditional_t<is_dynamic_array_v<UInt32>,
                                                            enoki::PCG32<UInt32, 1>,
                                                            enoki::PCG32<UInt32>>;

/**
 * \brief Generate fast and reasonably good pseudorandom numbers using the
 * Tiny Encryption Algorithm (TEA) by David Wheeler and Roger Needham.
 *
 * For details, refer to "GPU Random Numbers via the Tiny Encryption Algorithm"
 * by Fahad Zafar, Marc Olano, and Aaron Curtis.
 *
 * \param v0
 *     First input value to be encrypted (could be the sample index)
 * \param v1
 *     Second input value to be encrypted (e.g. the requested random number dimension)
 * \param rounds
 *     How many rounds should be executed? The default for random number
 *     generation is 4.
 * \return
 *     A uniformly distributed 32-bit integer
 */

template <typename UInt32>
UInt32 sample_tea_32(UInt32 v0, UInt32 v1, int rounds = 4) {
    UInt32 sum = 0;

    ENOKI_NOUNROLL for (int i = 0; i < rounds; ++i) {
        sum += 0x9e3779b9;
        v0 += (sl<4>(v1) + 0xa341316c) ^ (v1 + sum) ^ (sr<5>(v1) + 0xc8013ea4);
        v1 += (sl<4>(v0) + 0xad90777d) ^ (v0 + sum) ^ (sr<5>(v0) + 0x7e95761e);
    }

    return v1;
}

/**
 * \brief Generate fast and reasonably good pseudorandom numbers using the
 * Tiny Encryption Algorithm (TEA) by David Wheeler and Roger Needham.
 *
 * For details, refer to "GPU Random Numbers via the Tiny Encryption Algorithm"
 * by Fahad Zafar, Marc Olano, and Aaron Curtis.
 *
 * \param v0
 *     First input value to be encrypted (could be the sample index)
 * \param v1
 *     Second input value to be encrypted (e.g. the requested random number dimension)
 * \param rounds
 *     How many rounds should be executed? The default for random number
 *     generation is 4.
 * \return
 *     A uniformly distributed 64-bit integer
 */

template <typename UInt32>
uint64_array_t<UInt32> sample_tea_64(UInt32 v0, UInt32 v1, int rounds = 4) {
    UInt32 sum = 0;

    ENOKI_NOUNROLL for (int i = 0; i < rounds; ++i) {
        sum += 0x9e3779b9;
        v0 += (sl<4>(v1) + 0xa341316c) ^ (v1 + sum) ^ (sr<5>(v1) + 0xc8013ea4);
        v1 += (sl<4>(v0) + 0xad90777d) ^ (v0 + sum) ^ (sr<5>(v0) + 0x7e95761e);
    }

    return uint64_array_t<UInt32>(v0) + sl<32>(uint64_array_t<UInt32>(v1));
}


/**
 * \brief Generate fast and reasonably good pseudorandom numbers using the
 * Tiny Encryption Algorithm (TEA) by David Wheeler and Roger Needham.
 *
 * This function uses \ref sample_tea to return single precision floating point
 * numbers on the interval <tt>[0, 1)</tt>
 *
 * \param v0
 *     First input value to be encrypted (could be the sample index)
 * \param v1
 *     Second input value to be encrypted (e.g. the requested random number dimension)
 * \param rounds
 *     How many rounds should be executed? The default for random number
 *     generation is 4.
 * \return
 *     A uniformly distributed floating point number on the interval <tt>[0, 1)</tt>
 */
template <typename UInt32>
float32_array_t<UInt32> sample_tea_float32(UInt32 v0, UInt32 v1, int rounds = 4) {
    return reinterpret_array<float32_array_t<UInt32>>(
        sr<9>(sample_tea_32(v0, v1, rounds)) | 0x3f800000u) - 1.f;
}

/**
 * \brief Generate fast and reasonably good pseudorandom numbers using the
 * Tiny Encryption Algorithm (TEA) by David Wheeler and Roger Needham.
 *
 * This function uses \ref sample_tea to return double precision floating point
 * numbers on the interval <tt>[0, 1)</tt>
 *
 * \param v0
 *     First input value to be encrypted (could be the sample index)
 * \param v1
 *     Second input value to be encrypted (e.g. the requested random number dimension)
 * \param rounds
 *     How many rounds should be executed? The default for random number
 *     generation is 4.
 * \return
 *     A uniformly distributed floating point number on the interval <tt>[0, 1)</tt>
 */

template <typename UInt32>
float64_array_t<UInt32> sample_tea_float64(UInt32 v0, UInt32 v1, int rounds = 4) {
    return reinterpret_array<float64_array_t<UInt32>>(
        sr<12>(sample_tea_64(v0, v1, rounds)) | 0x3ff0000000000000ull) - 1.0;
}


/// Alias to \ref sample_tea_float32 or \ref sample_tea_float64 based given type size
template <typename UInt>
auto sample_tea_float(UInt v0, UInt v1, int rounds = 4) {
    if constexpr(std::is_same_v<scalar_t<UInt>, uint32_t>)
        return sample_tea_float32(v0, v1, rounds);
    else
        return sample_tea_float64(v0, v1, rounds);
}

/// Forward \ref next_float call to PCG32 random generator based given type size
template <typename Float, typename PCG32>
MTS_INLINE Float next_float(PCG32 *rng, mask_t<Float> active = true) {
    if constexpr (is_double_v<scalar_t<Float>>)
        return rng->next_float64(active);
    else
        return rng->next_float32(active);
}

/**
 * \brief Generate pseudorandom permutation vector using a shuffling network and the
 * \ref sample_tea function. This algorithm has a O(log2(sample_count)) complexity but
 * only supports permutation vectors whose lengths are a power of 2.
 *
 * \param index
 *     Input index to be mapped
 * \param sample_count
 *     Length of the permutation vector
 * \param seed
 *     Seed value used as second input to the Tiny Encryption Algorithm. Can be used to
 *     generate different permutation vectors.
 * \param rounds
 *     How many rounds should be executed by the Tiny Encryption Algorithm? The default is 2.
 * \return
 *     The index corresponding to the input index in the pseudorandom permutation vector.
 */

template <typename UInt32>
UInt32 permute(UInt32 index, uint32_t sample_count, UInt32 seed, int rounds = 2) {
    uint32_t  n = log2i(sample_count);
    Assert((1 << n) == sample_count, "sample_count should be a power of 2");

    for (uint32_t  level = 0; level < n; ++level) {
        UInt32 bit = UInt32(1 << level);
        // Take a random integer indentical for indices that might be swapped at this level
        UInt32 rand = sample_tea_32(index | bit, seed, rounds);
        masked(index, eq(rand & bit, bit)) = index ^ bit;
    }

    return index;
}

/**
 * \brief Generate pseudorandom permutation vector using the algorithm described in Pixar's
 * technical memo "Correlated Multi-Jittered Sampling":
 *
 *     https://graphics.pixar.com/library/MultiJitteredSampling/
 *
 *  Unlike \ref permute, this function supports permutation vectors of any length.
 *
 * \param index
 *     Input index to be mapped
 * \param sample_count
 *     Length of the permutation vector
 * \param seed
 *     Seed value used as second input to the Tiny Encryption Algorithm. Can be used to
 *     generate different permutation vectors.
 * \return
 *     The index corresponding to the input index in the pseudorandom permutation vector.
 */

template <typename UInt32>
UInt32 kensler_permute(UInt32 index, uint32_t sample_count, UInt32 seed, mask_t<UInt32> active = true) {
    if (sample_count == 1)
        return zero<UInt32>(slices(index));

    UInt32 w = sample_count - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;

    mask_t<UInt32> invalid = true;
    UInt32 tmp;
    do {
        tmp = index;
        tmp ^= seed;
        tmp *= 0xe170893d;
        tmp ^= seed >> 16;
        tmp ^= (tmp & w) >> 4;
        tmp ^= seed >> 8;
        tmp *= 0x0929eb3f;
        tmp ^= seed >> 23;
        tmp ^= (tmp & w) >> 1;
        tmp *= 1 | seed >> 27;
        tmp *= 0x6935fa69;
        tmp ^= (tmp & w) >> 11;
        tmp *= 0x74dcb303;
        tmp ^= (tmp & w) >> 2;
        tmp *= 0x9e501cc3;
        tmp ^= (tmp & w) >> 2;
        tmp *= 0xc860a3df;
        tmp &= w;
        tmp ^= tmp >> 5;
        masked(index, invalid) = tmp;
        invalid = (index >= sample_count);
    } while (any(active && invalid));

    return (index + seed) % sample_count;
}

NAMESPACE_END(mitsuba)
