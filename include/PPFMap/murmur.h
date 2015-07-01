//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.

/**
 *  This code was obtained from:
 *  https://code.google.com/p/smhasher/
 *
 *  and adapted to cuda and PPF encoding.
 */

#include <cuda_runtime.h>

/** \brief Bit-wise rotation of a 32-bit long word to the left.
 *  \param[in] x Word to rotate to the left.
 *  \param[in] r Magnitude of rotation.
 *  \return The rotated word.
 */
__host__ __device__ 
static inline uint32_t rotl32(uint32_t x, int8_t r) {
  return (x << r) | (x >> (32 - r));
}


/** \brief Murmur hashing for PPF Features into unsigned 32 bit integer.
 *
 *  This function is obtained from https://code.google.com/p/smhasher/. 
 *  Corresponds to the MurmurHash3_x86_32 function. This is basically the 
 *  result of fixing the length of the key to 16 (4 * unit32_t) and the initial 
 *  seed to 42.
 *
 *  \param[in] ppf Array containing the 4 components of a PPF feature.
 *  \return 32-bit unsigned integer hash of the feature.
 */
__host__ __device__
inline uint32_t murmurppf(const uint32_t ppf[4]) {
    uint32_t h1 = 42;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    for (int i = 0; i < 4; i++) {
        uint32_t k1 = ppf[i];

        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = rotl32(h1, 13); 
        h1 = h1 * 5 + 0xe6546b64;
    }

    h1 ^= 16;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    return h1;
}
