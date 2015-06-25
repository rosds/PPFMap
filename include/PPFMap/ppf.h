#ifndef PPFMAP_PPF_HH__
#define PPFMAP_PPF_HH__

#include <iostream>
#include <stdint.h>
#include <cuda_runtime.h>
#include <PPFMap/utils.h>

namespace ppfmap {

class PPFFeature {
public:
    /** \brief Empty constructor.
     *  
     *  Initializes each feature component to 0.
     */
    PPFFeature () : f1(0.0f), f2(0.0f), f3(0.0f), f4(0.0f) {}

    template <typename PointT, typename NormalT>
    PPFFeature(const PointT& p1, const NormalT& n1,
               const PointT& p2, const NormalT& n2);
    
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

    friend std::ostream& operator<<(std::ostream& os, const PPFFeature& f) {
        os << "[ " << f.f1 << " | " << f.f2 << " | " << f.f3 << " | " << f.f4 << " ]";
        return os;
    }

private:
    template <typename PointT, typename NormalT>
    void computePPFFeature(const PointT& p1, const NormalT& n1, const PointT& p2, const NormalT& n2);
};

#include <PPFMap/impl/ppf.hpp>

} // namespace ppfmap

#endif // PPFMAP_PPF_HH__
