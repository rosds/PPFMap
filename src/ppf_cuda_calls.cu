#include <thrust/host_vector.h>
#include <PPFMap/ppf_cuda_calls.h>

ppfmap::Map::Ptr 
ppfmap::cuda::setPPFMap(const float3 *points, 
                        const float3 *normals,
                        const size_t n,
                        const float disc_dist,
                        const float disc_angle) {

    thrust::host_vector<float3> h_points(points, points + n);
    thrust::host_vector<float3> h_normals(normals, normals + n);

    return boost::shared_ptr<Map>(new Map(h_points, h_normals, disc_dist, disc_angle));
}
