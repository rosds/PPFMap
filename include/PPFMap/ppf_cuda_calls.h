#ifndef PPFMAP_PPF_CUDA_CALLS_HH__
#define PPFMAP_PPF_CUDA_CALLS_HH__

#include <PPFMap/cuda_map.h>

namespace ppfmap {

namespace cuda {

    ppfmap::Map::Ptr setPPFMap(const float3 *points, 
                               const float3 *normals,
                               const size_t n,
                               const float disc_dist,
                               const float disc_angle);

} // namespace cuda

} // namespace ppfmap

#endif // PPFMAP_PPF_CUDA_CALLS_HH__
