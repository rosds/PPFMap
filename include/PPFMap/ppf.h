#ifndef PPFMAP_PPF_HH__
#define PPFMAP_PPF_HH__

#include <stdint.h>
#include <cuda_runtime.h>
#include <PPFMap/utils.h>

namespace ppfmap {

struct PPFFeature {
    /** \brief Empty constructor.
     */
    PPFFeature () : f1(0.0f), f2(0.0f), f3(0.0f), f4(0.0f) {}

    PPFFeature(const float3& p1, const float3& n1,
               const float3& p2, const float3& n2);

    virtual ~PPFFeature () {}

    union {
        float f[4];
        struct {
            float f1;
            float f2;
            float f3;
            float f4;
        };
    };

    inline uint32_t hash(const float disc_dist, const float disc_angle) {
        uint32_t d1 = static_cast<uint32_t>(f1 / disc_dist);
        uint32_t d2 = static_cast<uint32_t>(f2 / disc_angle);
        uint32_t d3 = static_cast<uint32_t>(f3 / disc_angle);
        uint32_t d4 = static_cast<uint32_t>(f4 / disc_angle);
        return d1 << 24 | d2 << 16 | d3 << 8 | d4;
    }
};

} // namespace ppfmap

#endif // PPFMAP_PPF_HH__
