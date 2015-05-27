#ifndef PPFMAP_PPF_CUDA_CALLS_HH__
#define PPFMAP_PPF_CUDA_CALLS_HH__

#include <PPFMap/Map.h>

namespace ppfmap {

namespace cuda {

    ppfmap::Map::Ptr setPPFMap(const float *points, 
                               const float *normals,
                               const size_t n,
                               const float disc_dist,
                               const float disc_angle);

} // namespace cuda

} // namespace ppfmap

#endif // PPFMAP_PPF_CUDA_CALLS_HH__
